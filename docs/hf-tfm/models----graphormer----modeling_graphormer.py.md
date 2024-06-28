# `.\models\graphormer\modeling_graphormer.py`

```py
# coding=utf-8
# Copyright 2022 Microsoft, clefourrier The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Graphormer model."""

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

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "graphormer-base-pcqm4mv1"
_CONFIG_FOR_DOC = "GraphormerConfig"

# 预训练模型存档列表
GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "clefourrier/graphormer-base-pcqm4mv1",
    "clefourrier/graphormer-base-pcqm4mv2",
    # 查看所有 Graphormer 模型的列表：https://huggingface.co/models?filter=graphormer
]

def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """

    # 如果没有量化噪声，则不注册钩子
    if p <= 0:
        return module

    # 只支持以下类型的模块
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # 检查 module.weight 的维度是否符合 block_size
    is_conv = module.weight.ndim == 4

    # 2D 矩阵
    # 如果不是卷积层
    if not is_conv:
        # 检查输入特征是否是块大小的倍数，如果不是则抛出断言错误
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 如果是卷积层
    else:
        # 对于 1x1 卷积
        if module.kernel_size == (1, 1):
            # 检查输入通道数是否是块大小的倍数，如果不是则抛出断言错误
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # 对于常规卷积
        else:
            # 计算卷积核大小
            k = module.kernel_size[0] * module.kernel_size[1]
            # 检查卷积核大小是否是块大小的倍数，如果不是则抛出断言错误
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    # 定义一个前向预处理钩子函数
    def _forward_pre_hook(mod, input):
        # 如果处于训练模式
        if mod.training:
            # 如果不是卷积层
            if not is_conv:
                # 获取权重和大小信息
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # 创建一个用于掩码的零张量，并根据概率 p 进行 Bernoulli 采样
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # 获取权重和大小信息
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # 根据卷积核大小创建一个用于掩码的零张量，并根据概率 p 进行 Bernoulli 采样
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

            # 将掩码转换为布尔型张量，以便 TorchScript 中的兼容性
            mask = mask.to(torch.bool)
            # 计算权重的缩放因子并应用掩码
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    # 注册前向预处理钩子函数
    module.register_forward_pre_hook(_forward_pre_hook)
    # 返回处理后的模块
    return module
# 定义一个继承自 `nn.ModuleList` 的自定义模块 `LayerDropModuleList`
class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    # 初始化方法，接收概率参数 p 和模块列表 modules
    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None):
        # 调用父类的初始化方法，将 modules 传入父类构造函数
        super().__init__(modules)
        # 存储概率参数 p
        self.p = p

    # 迭代器方法，返回一个迭代器，迭代模块列表中的每个模块
    def __iter__(self) -> Iterator[nn.Module]:
        # 创建一个与模块列表长度相同的随机概率张量
        dropout_probs = torch.empty(len(self)).uniform_()
        # 遍历模块列表中的每个模块和对应的随机概率
        for i, m in enumerate(super().__iter__()):
            # 如果不在训练模式下或者随机概率大于阈值 p，则保留该模块
            if not self.training or (dropout_probs[i] > self.p):
                yield m


# 定义一个用于计算图中节点特征的模块 `GraphormerGraphNodeFeature`
class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    # 初始化方法，接收一个 `GraphormerConfig` 类型的配置参数 config
    def __init__(self, config: GraphormerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 存储配置参数中的注意力头数和原子数
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        # 初始化原子编码器，输入维度为原子数加一，输出维度为隐藏大小，设置了填充索引
        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化入度编码器，输入维度为入度数，输出维度为隐藏大小，设置了填充索引
        self.in_degree_encoder = nn.Embedding(
            config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id
        )
        # 初始化出度编码器，输入维度为出度数，输出维度为隐藏大小，设置了填充索引
        self.out_degree_encoder = nn.Embedding(
            config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id
        )

        # 初始化图标记编码器，固定为 1 的嵌入，输出维度为隐藏大小
        self.graph_token = nn.Embedding(1, config.hidden_size)

    # 前向传播方法，接收输入节点、入度和出度张量，返回节点特征张量
    def forward(
        self,
        input_nodes: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
    ) -> torch.Tensor:
        # 获取输入节点张量的维度信息，n_graph 表示图的数量，n_node 表示每个图中节点的数量
        n_graph, n_node = input_nodes.size()[:2]

        # 计算节点特征，包括原子编码器的求和、入度编码器和出度编码器
        node_feature = (
            self.atom_encoder(input_nodes).sum(dim=-2)  # [n_graph, n_node, n_hidden]
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        # 计算图标记特征，使用图标记编码器的权重张量重复 n_graph 次
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # 拼接图标记特征和节点特征，沿着第一个维度拼接
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # 返回拼接后的图节点特征张量
        return graph_node_feature


# 定义一个用于计算每个注意力头的注意力偏置的模块 `GraphormerGraphAttnBias`
class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    # 初始化方法，无需额外参数
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 此处无需定义前向传播方法，因为此模块仅用于计算注意力偏置
    # 初始化函数，接受一个 GraphormerConfig 类型的参数 config
    def __init__(self, config: GraphormerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置头注意力的数量为配置参数中的 num_attention_heads
        self.num_heads = config.num_attention_heads
        # 设置多跳最大距离为配置参数中的 multi_hop_max_dist
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # 创建一个边特征编码器，使用 nn.Embedding 来表示边特征的组合
        # config.num_edges + 1 表示边特征的数量，config.num_attention_heads 表示每个边特征的维度，padding_idx=0 表示填充索引为0
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)

        # 设置边类型为配置参数中的 edge_type
        self.edge_type = config.edge_type
        # 如果边类型是 "multi_hop"，则创建一个边距离编码器
        if self.edge_type == "multi_hop":
            # 使用 nn.Embedding 创建边距离编码器，大小为 config.num_edge_dis * config.num_attention_heads * config.num_attention_heads
            # 输出维度为1
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis * config.num_attention_heads * config.num_attention_heads,
                1,
            )

        # 创建空间位置编码器，使用 nn.Embedding 表示空间位置
        # config.num_spatial 表示空间位置的数量，config.num_attention_heads 表示每个位置的维度，padding_idx=0 表示填充索引为0
        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)

        # 创建图令牌的虚拟距离编码器，使用 nn.Embedding 表示虚拟距离
        # 大小为1，config.num_attention_heads 表示每个虚拟距离的维度
        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    # 前向传播函数，接受多个张量作为输入参数
    def forward(
        self,
        input_nodes: torch.LongTensor,
        attn_bias: torch.Tensor,
        spatial_pos: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        ...
        # 返回类型声明为 torch.Tensor
        n_graph, n_node = input_nodes.size()[:2]
        # 复制注意力偏置张量
        graph_attn_bias = attn_bias.clone()
        # 在第1维度上增加维度，并重复 num_heads 次，扩展为四维张量
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # 空间位置偏置处理
        # 调用 spatial_pos_encoder 处理 spatial_pos，并进行维度置换
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # 将 spatial_pos_bias 加到 graph_attn_bias 的对应位置上
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # 在这里重置空间位置
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # 将 t 加到 graph_attn_bias 的对应位置上
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 边特征处理
        if self.edge_type == "multi_hop":
            # 复制 spatial_pos，并将值为0的元素设为1
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # 将大于1的元素减1，同时限制到 multi_hop_max_dist 的范围内
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
            # 对 input_edges 进行边编码处理，并在倒数第二维求均值
            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            # 对 edge_input_flat 进行形状重塑和矩阵乘法操作
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            # 对 input_edges 进行形状重塑和维度置换
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(
                1, 2, 3, 0, 4
            )
            input_edges = (input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # 对 attn_edge_type 进行边编码处理，并在倒数第二维求均值，并进行维度置换
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        # 将 input_edges 加到 graph_attn_bias 的对应位置上
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        # 将 attn_bias 增加一个维度后加到 graph_attn_bias 上，进行重置
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        # 返回 graph_attn_bias 结果
        return graph_attn_bias
    def reset_parameters(self):
        # 如果查询、键和值的维度相同，则使用缩放的初始化方法
        if self.qkv_same_dim:
            # 使用缩放的初始化方法对 k_proj, v_proj, q_proj 的权重进行初始化
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            # 使用普通的 xavier_uniform 初始化方法对 k_proj, v_proj, q_proj 的权重进行初始化
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        # 使用 xavier_uniform 初始化方法对 out_proj 的权重进行初始化
        nn.init.xavier_uniform_(self.out_proj.weight)
        # 如果 out_proj 的偏置存在，则将其初始化为常数 0.0
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    # 定义了一个方法 `forward`，用于执行模型的前向传播
    def forward(
        self,
        query: torch.LongTensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ):
        # apply_sparse_mask 方法，用于对注意力权重矩阵进行稀疏掩码处理
        def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
            # 直接返回传入的注意力权重矩阵
            return attn_weights
class GraphormerGraphEncoderLayer(nn.Module):
    # Graphormer 图编码器层的定义，继承自 nn.Module
    def __init__(self, config: GraphormerConfig) -> None:
        super().__init__()

        # 初始化参数
        self.embedding_dim = config.embedding_dim  # 嵌入维度
        self.num_attention_heads = config.num_attention_heads  # 注意力头的数量
        self.q_noise = config.q_noise  # 量化噪声
        self.qn_block_size = config.qn_block_size  # 量化块大小
        self.pre_layernorm = config.pre_layernorm  # 是否使用层标准化前置

        # 初始化 Dropout 模块
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        # 初始化激活函数 Dropout 模块
        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)

        # 初始化激活函数
        self.activation_fn = ACT2FN[config.activation_fn]

        # 初始化自注意力层
        self.self_attn = GraphormerMultiheadAttention(config)

        # 自注意力层后的层标准化
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        # 构建第一个全连接层
        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # 构建第二个全连接层
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # 最终层的层标准化
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        # 构建带有量化噪声的全连接层
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        input_nodes: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        ):
        # 前向传播函数定义
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        定义函数的返回类型为一个包含两个元素的元组，第一个是 torch.Tensor 类型，第二个是可选的 torch.Tensor 类型。
        """
        residual = input_nodes
        # 如果配置为在 self-attention/ffn 模块之前应用 LayerNorm，则对输入进行 LayerNorm
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        # 调用 self-attention 模块进行计算
        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        # 对输出结果进行 dropout 处理
        input_nodes = self.dropout_module(input_nodes)
        # 将残差连接到当前的输出上
        input_nodes = residual + input_nodes
        # 如果配置为在 self-attention/ffn 模块之后应用 LayerNorm，则对输出进行 LayerNorm
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        # 如果配置为在最终层之前应用 LayerNorm，则对输入进行 LayerNorm
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        # 应用激活函数到第一个全连接层的输出上
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        # 对第一个全连接层的输出进行 dropout 处理
        input_nodes = self.activation_dropout_module(input_nodes)
        # 通过第二个全连接层进行计算
        input_nodes = self.fc2(input_nodes)
        # 对第二个全连接层的输出进行 dropout 处理
        input_nodes = self.dropout_module(input_nodes)
        # 将残差连接到当前的输出上
        input_nodes = residual + input_nodes
        # 如果配置为在最终层之后应用 LayerNorm，则对输出进行 LayerNorm
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        # 返回计算结果及 self-attention 的注意力权重（如果有的话）
        return input_nodes, attn
class GraphormerGraphEncoder(nn.Module):
    # 定义 Graphormer 图编码器模型
    def __init__(self, config: GraphormerConfig):
        super().__init__()

        # 定义模型的 dropout 模块，用于随机失活
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        
        # 设置层级随机失活率
        self.layerdrop = config.layerdrop
        
        # 设置嵌入维度
        self.embedding_dim = config.embedding_dim
        
        # 是否应用 Graphormer 初始化
        self.apply_graphormer_init = config.apply_graphormer_init
        
        # 是否支持跟踪
        self.traceable = config.traceable

        # 初始化 Graphormer 图节点特征
        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        
        # 初始化 Graphormer 图注意力偏置
        self.graph_attn_bias = GraphormerGraphAttnBias(config)

        # 设置嵌入缩放
        self.embed_scale = config.embed_scale

        # 如果配置中有量化噪声，则初始化量化噪声模块
        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        # 根据配置决定是否使用 Encoder 前归一化
        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        # 如果配置中有预归一化选项，则初始化最终的层级归一化
        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        # 如果配置中定义了层级随机失活率，则创建相应的层级列表
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        # 根据配置中的隐藏层数量，扩展层级列表，每层使用 Graphormer 图编码器层进行初始化
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 在构建模型后，根据配置决定是否应用模型参数的初始化
        # 如果配置中冻结嵌入，则抛出未实现异常
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        # 冻结指定数量的转换层参数
        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    # 前向传播函数定义，接受多个输入张量和可选参数
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
        # 定义函数签名和返回类型注释
        ) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        # 计算填充掩码。这对多头注意力很重要
        data_x = input_nodes
        # 获取数据的图和节点数量
        n_graph, n_node = data_x.size()[:2]
        # 创建填充掩码，找出哪些位置是填充的
        padding_mask = (data_x[:, :, 0]).eq(0)
        # 创建一个新的填充掩码，用于CLS位置的特殊标记
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        # 将CLS掩码和普通掩码拼接在一起
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        # 计算注意力偏置
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)

        # 如果存在令牌嵌入，则使用令牌嵌入作为输入节点特征
        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            # 否则，通过计算节点特征得到输入节点
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)

        # 如果存在扰动，则添加扰动到输入节点
        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        # 如果存在嵌入缩放因子，则对输入节点进行缩放
        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        # 如果存在量化噪声，则应用量化噪声到输入节点
        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        # 如果存在嵌入层规范化，则对输入节点进行规范化
        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        # 对输入节点应用丢弃模块，以防止过拟合
        input_nodes = self.dropout_module(input_nodes)

        # 转置输入节点，以适应模型需求
        input_nodes = input_nodes.transpose(0, 1)

        # 初始化内部状态列表
        inner_states = []
        # 如果不仅需最后一个状态，则将当前输入节点添加到内部状态列表
        if not last_state_only:
            inner_states.append(input_nodes)

        # 遍历所有层，依次进行计算和更新
        for layer in self.layers:
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            # 如果不仅需最后一个状态，则将当前输入节点添加到内部状态列表
            if not last_state_only:
                inner_states.append(input_nodes)

        # 提取图表示，通常为第一个节点的输出
        graph_rep = input_nodes[0, :, :]

        # 如果仅需最后一个状态，则重置内部状态列表为仅包含最后一个状态
        if last_state_only:
            inner_states = [input_nodes]

        # 如果支持追踪，则返回内部状态的堆栈和图表示
        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            # 否则，仅返回内部状态和图表示
            return inner_states, graph_rep
    # 定义一个 Graphormer 解码器的头部模块，继承自 nn.Module
    class GraphormerDecoderHead(nn.Module):
        def __init__(self, embedding_dim: int, num_classes: int):
            super().__init__()
            """num_classes should be 1 for regression, or the number of classes for classification"""
            # 初始化一个学习偏置参数，用于输出层的线性变换
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            # 定义一个线性分类器，输入维度为 embedding_dim，输出维度为 num_classes，不使用偏置项
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
            self.num_classes = num_classes

        def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
            # 将输入节点 input_nodes 经过分类器进行线性变换
            input_nodes = self.classifier(input_nodes)
            # 加上学习的偏置参数 lm_output_learned_bias
            input_nodes = input_nodes + self.lm_output_learned_bias
            # 返回处理后的节点数据
            return input_nodes


    class GraphormerPreTrainedModel(PreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """

        # 指定配置类为 GraphormerConfig
        config_class = GraphormerConfig
        # 基础模型的前缀名称为 "graphormer"
        base_model_prefix = "graphormer"
        # 主要输入节点的名称为 "input_nodes"
        main_input_name_nodes = "input_nodes"
        # 主要输入边的名称为 "input_edges"

        def normal_(self, data: torch.Tensor):
            # 使用 FSDP（Fully Sharded Data Parallel）时，模块参数会在 CUDA 上，因此将它们转回 CPU
            # 以确保随机数生成器在有无 FSDP 时保持一致性
            data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

        def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
            """
            Initialize the weights specific to the Graphormer Model.
            """
            # 根据模块的类型初始化 Graphormer 模型特定的权重
            if isinstance(module, nn.Linear):
                # 初始化线性层的权重
                self.normal_(module.weight.data)
                # 如果有偏置项，则将其置零
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                # 初始化嵌入层的权重
                self.normal_(module.weight.data)
                # 如果有填充索引，则将填充索引处的权重置零
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, GraphormerMultiheadAttention):
                # 初始化 Graphormer 多头注意力机制中的投影权重
                self.normal_(module.q_proj.weight.data)
                self.normal_(module.k_proj.weight.data)
                self.normal_(module.v_proj.weight.data)

        def _init_weights(
            self,
            module: Union[
                nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, GraphormerMultiheadAttention, GraphormerGraphEncoder
            ],
        """
        初始化模型的权重
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或卷积层，初始化权重为正态分布，均值为0，标准差为0.02
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                # 如果存在偏置项，将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，初始化权重为正态分布，均值为0，标准差为0.02
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # 如果设置了padding_idx，将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphormerMultiheadAttention):
            # 如果是自定义的多头注意力层，分别初始化查询、键、值投影的权重
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            # 调用重置参数的方法
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            # 如果是层归一化层，初始化偏置为零，权重为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphormerGraphEncoder):
            # 如果是自定义的图编码器，根据设置决定是否应用初始化方法
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)
        elif isinstance(module, nn.LayerNorm):
            # 再次检查是否是层归一化层，确保偏置为零，权重为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__(config)
        self.max_nodes = config.max_nodes  # 初始化最大节点数

        self.graph_encoder = GraphormerGraphEncoder(config)  # 初始化图编码器

        self.share_input_output_embed = config.share_input_output_embed  # 是否共享输入输出的嵌入
        self.lm_output_learned_bias = None  # 学习到的偏置参数为 None

        # Fine-tuning时设置为True
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)  # 线性变换权重
        self.activation_fn = ACT2FN[config.activation_fn]  # 激活函数
        self.layer_norm = nn.LayerNorm(config.embedding_dim)  # 归一化层

        self.post_init()  # 调用后初始化方法

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))  # 重置输出层参数为学习到的偏置

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inner_states, graph_rep = self.graph_encoder(  # 调用图编码器进行前向传播
            input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb
        )

        # 取最后一个内部状态，然后反转批次和图长度
        input_nodes = inner_states[-1].transpose(0, 1)

        # 仅投影掩码的标记
        if masked_tokens is not None:
            raise NotImplementedError  # 抛出未实现错误

        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))  # 执行线性变换和激活函数后进行归一化

        # 投影回词汇表的大小
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            input_nodes = torch.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)  # 返回模型输出

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes  # 返回最大节点数
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
        super().__init__(config)
        # 初始化 GraphormerForGraphClassification 类的实例
        self.encoder = GraphormerModel(config)
        # 设置嵌入维度
        self.embedding_dim = config.embedding_dim
        # 设置类别数量
        self.num_classes = config.num_classes
        # 初始化分类器
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        # 表示这个模型是编码器-解码器结构
        self.is_encoder_decoder = True

        # 初始化权重并进行最终处理
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据传递给编码器
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
        # 获取编码器的输出和隐藏状态
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        # 将编码器的输出传递给分类器
        head_outputs = self.classifier(outputs)
        # 取出分类器的 logits（对数几率）
        logits = head_outputs[:, 0, :].contiguous()

        loss = None
        if labels is not None:
            # 创建一个标签的掩码，用于处理缺失的标签数据
            mask = ~torch.isnan(labels)

            if self.num_classes == 1:  # regression（回归）
                # 如果是回归任务，使用均方误差损失函数
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:  # One task classification（单一任务分类）
                # 如果是单一任务分类，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # Binary multi-task classification（二进制多任务分类）
                # 如果是二进制多任务分类，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        # 如果不要求返回字典，则返回损失、logits 和隐藏状态的元组
        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        # 如果要求返回字典，则返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)
```