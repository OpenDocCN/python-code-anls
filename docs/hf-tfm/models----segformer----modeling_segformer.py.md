# `.\models\segformer\modeling_segformer.py`

```
# coding=utf-8
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SegFormer model."""

# 导入所需的库和模块
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入各种辅助函数和类
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SegformerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型的存档列表
SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # See all SegFormer models at https://huggingface.co/models?filter=segformer
]

# ImageClassifierOutput 的子类，用于图像分类模型的输出
class SegFormerImageClassifierOutput(ImageClassifierOutput):
    """
    Base class for outputs of image classification models.
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果`config.num_labels==1`则为回归）损失。
            Loss for classification (or regression if `config.num_labels==1`).
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果`config.num_labels==1`则为回归）得分（SoftMax 之前）。
            Scores for classification (or regression if `config.num_labels==1`), before SoftMax.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 的元组（如果模型有嵌入层，则包括嵌入层输出，以及每个阶段的输出），
            形状为 `(batch_size, num_channels, height, width)`。
            模型在每个阶段输出的隐藏状态（也称为特征图）。
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer,
            plus one for the output of each stage), with shape `(batch_size, num_channels, height, width)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 的元组（每个层一个），形状为 `(batch_size, num_heads, patch_size, sequence_length)`。
            自注意力机制中注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
            Tuple of `torch.FloatTensor` (one for each layer), with shape `(batch_size, num_heads, patch_size, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 处理不同维度的张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Segformer
class SegformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数来执行 drop path 操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        # 使用卷积层构建重叠的补丁嵌入
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        # 计算嵌入，然后重塑形状以便传递给 Transformer 层
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""
    # 初始化函数，用于创建一个注意力机制模型
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        # 调用父类初始化方法
        super().__init__()
        # 将隐藏大小和注意力头数保存到对象属性中
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # 检查隐藏大小是否能被注意力头数整除，否则抛出数值错误
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        # 计算每个注意力头的大小和所有注意力头的总大小
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建三个线性变换层，用于生成查询、键和值
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        # 创建一个丢弃层，用于注意力概率的丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果序列缩减比例大于1，则创建一个卷积层和层归一化层
        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    # 将隐藏状态重塑为注意力分数计算所需的形状
    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    # 前向传播函数，定义了模型如何处理输入并生成输出
    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
        ):
            # 通过对隐藏状态进行查询操作并转置以备使用
            query_layer = self.transpose_for_scores(self.query(hidden_states))

            # 如果序列压缩比大于1
            if self.sr_ratio > 1:
                batch_size, seq_len, num_channels = hidden_states.shape
                # 重新组织张量形状为(batch_size, num_channels, height, width)
                hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
                # 应用序列压缩操作
                hidden_states = self.sr(hidden_states)
                # 将张量形状还原为(batch_size, seq_len, num_channels)
                hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
                # 应用层归一化
                hidden_states = self.layer_norm(hidden_states)

            # 通过对隐藏状态进行键操作并转置以备使用
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # 通过对隐藏状态进行值操作并转置以备使用
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # 计算“查询”和“键”之间的点积，得到原始注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 对注意力分数进行缩放
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # 将注意力分数归一化为概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # 使用dropout来随机丢弃一些token，以实现注意力机制
            attention_probs = self.dropout(attention_probs)

            # 计算加权和，得到上下文张量
            context_layer = torch.matmul(attention_probs, value_layer)

            # 将上下文张量进行维度变换，以适应后续操作
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            # 输出结果，根据需要是否包含注意力分数
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            # 返回计算结果
            return outputs
# 定义一个自定义的 PyTorch 模块，用于 Segformer 模型的自注意力机制输出层
class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        # 线性层，将隐藏状态映射到相同大小的空间
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Dropout 层，用于随机置零输入张量的元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # Dropout 操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 定义 Segformer 模型的注意力机制模块
class SegformerAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        # SegformerEfficientSelfAttention 自注意力模块的实例化
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # SegformerSelfOutput 自注意力输出层的实例化
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        # 用于存储被剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可剪枝的注意力头并获取索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # 调用自注意力模块的前向传播
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 通过输出层处理注意力输出和原始隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，添加注意力信息到输出元组中
        return outputs


# Segformer 模型的深度可分离卷积模块
class SegformerDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 深度可分离卷积层，用于处理输入的隐藏状态
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        # 获取输入隐藏状态的维度信息
        batch_size, seq_len, num_channels = hidden_states.shape
        # 调整隐藏状态的形状以便进行深度可分离卷积
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)  # 应用深度可分离卷积
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # 将结果展平并重塑
        return hidden_states


# Segformer 模型的混合前馈网络模块
class SegformerMixFFN(nn.Module):
    # 初始化函数，用于初始化一个自定义的神经网络模块
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        # 调用父类的初始化函数，确保正确地初始化神经网络模块
        super().__init__()
        # 如果未指定输出特征数，则默认与输入特征数相同
        out_features = out_features or in_features
        # 创建一个线性层，输入特征数为in_features，输出特征数为hidden_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 创建一个自定义的深度可分离卷积层
        self.dwconv = SegformerDWConv(hidden_features)
        # 根据配置文件中的隐藏激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个线性层，输入特征数为hidden_features，输出特征数为out_features
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 创建一个dropout层，使用配置文件中指定的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了数据在模型中如何流动的过程
    def forward(self, hidden_states, height, width):
        # 第一层线性变换，将输入特征向量映射到隐藏特征空间
        hidden_states = self.dense1(hidden_states)
        # 深度可分离卷积层的前向计算，处理输入特征的空间信息
        hidden_states = self.dwconv(hidden_states, height, width)
        # 使用配置文件中指定的中间激活函数对隐藏状态进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对隐藏状态应用dropout操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 第二层线性变换，将隐藏特征映射到输出特征空间
        hidden_states = self.dense2(hidden_states)
        # 再次应用dropout操作，增强模型的泛化能力
        hidden_states = self.dropout(hidden_states)
        # 返回最终的模型输出结果
        return hidden_states
class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        # Layer normalization applied to the input hidden states
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        
        # Self-attention mechanism specific to Segformer
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        
        # DropPath module for stochastic depth regularization
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Layer normalization applied after self-attention and before MLP
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        
        # Multi-layer perceptron (MLP) component of the Segformer layer
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # Apply layer normalization before feeding into self-attention
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # First residual connection with stochastic depth (DropPath)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        # Feed the output of the self-attention through the MLP
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # Second residual connection with stochastic depth (DropPath)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs  # Include layer output in the outputs tuple

        return outputs
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数 config 存储在对象的属性中
        self.config = config

        # stochastic depth decay rule
        # 根据 config 中的 drop_path_rate 参数生成一个随机深度衰减规则列表
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        # 遍历 num_encoder_blocks 次，创建 SegformerOverlapPatchEmbeddings 对象并加入列表中
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        # 将 embeddings 转换为 nn.ModuleList 类型，并赋值给 patch_embeddings 属性
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        # 遍历 num_encoder_blocks 次，创建 SegformerLayer 对象并加入列表中
        for i in range(config.num_encoder_blocks):
            # 每个块由多个层组成
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            # 根据 depths[i] 参数创建多个 SegformerLayer 层，并加入 layers 列表中
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            # 将 layers 转换为 nn.ModuleList 类型，并加入 blocks 列表中
            blocks.append(nn.ModuleList(layers))

        # 将 blocks 转换为 nn.ModuleList 类型，并赋值给 block 属性
        self.block = nn.ModuleList(blocks)

        # Layer norms
        # 根据 hidden_sizes[i] 创建多个 LayerNorm 层，并转换为 nn.ModuleList 类型，赋值给 layer_norm 属性
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    # 前向传播函数，接受像素值 pixel_values 和几个可选参数
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutput]:
        # 初始化用于存储所有隐藏状态的元组，如果不需要输出隐藏状态，则置为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有自注意力矩阵的元组，如果不需要输出注意力矩阵，则置为 None
        all_self_attentions = () if output_attentions else None

        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]

        # 将输入张量作为初始隐藏状态
        hidden_states = pixel_values
        # 遍历每个模块：嵌入层、块层、层归一化层
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # 第一步：获取补丁嵌入
            hidden_states, height, width = embedding_layer(hidden_states)
            # 第二步：将嵌入通过块层处理
            for i, blk in enumerate(block_layer):
                # 调用块层处理隐藏状态、高度、宽度以及是否输出注意力矩阵
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]  # 更新隐藏状态为块层输出的隐藏状态
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)  # 更新自注意力矩阵元组

            # 第三步：应用层归一化
            hidden_states = norm_layer(hidden_states)

            # 第四步：根据需要将隐藏状态重塑回 (batch_size, num_channels, height, width) 的形状
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

            # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出
        if not return_dict:
            # 返回包含非空值的元组（隐藏状态、所有隐藏状态、所有自注意力矩阵）
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # 返回 BaseModelOutput 类的实例，包含最终的隐藏状态、所有隐藏状态和所有自注意力矩阵
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 添加起始文档字符串和类注释，说明这是SegformerModel类，继承自SegformerPreTrainedModel类。
@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
class SegformerModel(SegformerPreTrainedModel):

    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    
    Attributes:
        config_class (SegformerConfig): Configuration class defining parameters for the model.
        base_model_prefix (str): Prefix used in naming the base model.
        main_input_name (str): Name of the main input expected by the model.
    """

    config_class = SegformerConfig
    base_model_prefix = "segformer"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights of the given module."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Initialize weights using normal distribution with mean 0 and standard deviation `initializer_range`
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # If bias exists, initialize it to zero
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights using normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # If padding index is specified, set corresponding embedding weights to zero
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization bias to zero and weight to one
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)
        # 初始化一个分层Transformer编码器，使用给定的配置参数

        # Initialize weights and apply final processing
        self.post_init()
        # 初始化权重并进行最终处理

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对模型的注意力头进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """
    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
    states) e.g. for ImageNet.
    """,
    SEGFORMER_START_DOCSTRING,
)
class SegformerForImageClassification(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 初始化 SegFormer 模型
        self.segformer = SegformerModel(config)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SegFormerImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 SegFormer 模型进行预测
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出的最后一层隐藏状态
        sequence_output = outputs[0]

        # 将最后一层隐藏状态转换为 (batch_size, height*width, hidden_size) 的形式
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # 如果需要重塑最后一个阶段的输出形状
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # 对每个样本进行全局平均池化
        sequence_output = sequence_output.mean(dim=1)

        # 使用分类器对平均池化后的特征进行分类预测
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 如果提供了标签
            if self.config.problem_type is None:
                # 根据标签数据类型和类别数设置问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                # 如果是回归问题，使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果是单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果是多标签分类问题，使用带logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则返回模型预测的输出和损失
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，返回 SegFormerImageClassifierOutput 类型的对象
        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        # 使用线性层将输入的维度 input_dim 转换为 config.decoder_hidden_size
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # 将 hidden_states 按照第二个维度展平，然后转置第一和第二维度
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # 使用 self.proj 进行线性变换
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 创建一个包含多个 SegformerMLP 的 ModuleList，用于将每个 encoder block 的通道维度统一到 config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # 实现原始实现的 ConvModule 的三个层
        # 使用 1x1 卷积层将输入通道数从 config.decoder_hidden_size * config.num_encoder_blocks 转换为 config.decoder_hidden_size
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        # 批量归一化层，归一化 config.decoder_hidden_size 个通道
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        # ReLU 激活函数
        self.activation = nn.ReLU()

        # 使用 config.classifier_dropout_prob 概率进行 Dropout
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 1x1 卷积层将 config.decoder_hidden_size 个通道转换为 config.num_labels 个通道
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config
    # 定义前向传播方法，接受编码器隐藏状态作为输入并返回预测的逻辑张量
    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 获取批量大小，这里假设输入的最后一个隐藏状态作为参考
        batch_size = encoder_hidden_states[-1].shape[0]

        # 初始化一个空元组，用于存储所有隐藏状态
        all_hidden_states = ()
        # 遍历编码器隐藏状态和线性层列表
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # 如果指定不重塑最后阶段并且编码器隐藏状态是3维的
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                # 计算高度和宽度，并重塑编码器隐藏状态的形状
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # 统一通道维度
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)  # 应用线性层到隐藏状态
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)  # 调换维度顺序
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)  # 重塑隐藏状态
            # 上采样
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            # 将当前处理后的隐藏状态添加到元组中
            all_hidden_states += (encoder_hidden_state,)

        # 将所有处理后的隐藏状态在通道维度上拼接并通过线性融合层处理
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)  # 批量归一化
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        hidden_states = self.dropout(hidden_states)  # 应用丢弃（dropout）操作

        # 最终的逻辑张量形状为 (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits
# 使用装饰器为类添加文档字符串，描述该类为基于 SegFormer 模型的语义分割模型，具有一个全MLP解码头部
# 可用于处理例如ADE20k和CityScapes数据集的任务
@add_start_docstrings(
    """SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.""",
    SEGFORMER_START_DOCSTRING,
)
# 定义 SegformerForSemanticSegmentation 类，继承自 SegformerPreTrainedModel 类
class SegformerForSemanticSegmentation(SegformerPreTrainedModel):
    
    # 初始化方法，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 SegformerModel 的实例，传入配置对象作为参数
        self.segformer = SegformerModel(config)
        # 创建 SegformerDecodeHead 的实例，传入配置对象作为参数
        self.decode_head = SegformerDecodeHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述其输入和输出
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串，指定输出类型为 SemanticSegmenterOutput，并指定配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    # forward 方法定义，接收多个输入参数，包括像素值、标签等信息
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```