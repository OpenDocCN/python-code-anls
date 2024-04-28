# `.\transformers\models\videomae\modeling_videomae.py`

```
# 这是一个 PyTorch 实现的 VideoMAE (masked autoencoder) 模型的代码
# 引入所需的库和类
# coding=utf-8
# Copyright 2022 Multimedia Computing Group, Nanjing University and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VideoMAE (masked autoencoder) model."""


import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig


# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文档中使用的配置和检查点
_CONFIG_FOR_DOC = "VideoMAEConfig"
_CHECKPOINT_FOR_DOC = "MCG-NJU/videomae-base"

# 列出支持的预训练模型
VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MCG-NJU/videomae-base",
    # See all VideoMAE models at https://huggingface.co/models?filter=videomae
]


# 定义 VideoMAEDecoderOutput 类，用于输出解码结果
@dataclass
class VideoMAEDecoderOutput(ModelOutput):
    """
    Class for VideoMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    # 定义 logits 变量，类型为 torch.FloatTensor，默认为 None
    logits: torch.FloatTensor = None
    # 定义 hidden_states 变量，类型为 Tuple[torch.FloatTensor] 或 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 变量，类型为 Tuple[torch.FloatTensor] 或 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 创建一个输出类，用于VideoMAEForPreTraining的输出，包含潜在的隐藏状态和注意力
@dataclass
class VideoMAEForPreTrainingOutput(ModelOutput):
    
    # 损失，形状为(1,)的FloatTensor，用于像素重构损失
    loss: Optional[torch.FloatTensor] = None
    
    # logits，形状为(batch_size,patch_size ** 2 * num_channels)的FloatTensor，用于像素重构的logits
    logits: torch.FloatTensor = None
    
    # hidden_states，类型为tuple(torch.FloatTensor)，可选项，当传入output_hidden_states=True或者config.output_hidden_states=True时返回
    # 包含模型每一层的隐藏状态，以及初始的嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # attentions，类型为tuple(torch.FloatTensor)，可选项，当传入output_attentions=True或者config.output_attentions=True时返回
    # 包含每一层的注意力权重，用于计算自注意力头部的加权平均
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# sin-cos位置编码函数
# 使用numpy的示例:
#   https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    
    # 用于计算每个位置的角度向量
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 创建n_position个位置的角度表
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    
    # 对角度表中的偶数列进行sin计算
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    
    # 对角度表中的奇数列进行cos计算
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # 将角度表转换为FloatTensor，增加一个维度
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VideoMAEEmbeddings(nn.Module):
    """
    构建patch和position的嵌入层
    """

    def __init__(self, config):
        super().__init__()

        # 创建patch嵌入层对象
        self.patch_embeddings = VideoMAEPatchEmbeddings(config)
        
        # 获取patch的数量
        self.num_patches = self.patch_embeddings.num_patches
        
        # 创建固定的sin-cos位置嵌入
        self.position_embeddings = get_sinusoid_encoding_table(self.num_patches, config.hidden_size)
        self.config = config
    # 定义神经网络的前向传播函数，接受像素数值和可见性掩码作为输入
    def forward(self, pixel_values, bool_masked_pos):
        # 创建分块嵌入
        embeddings = self.patch_embeddings(pixel_values)
    
        # 添加位置嵌入
        # 将位置嵌入型转为和嵌入张量相同的数据类型和设备，并复制独立张量
        embeddings = embeddings + self.position_embeddings.type_as(embeddings).to(embeddings.device).clone().detach()
    
        # 仅保留可见的分块
        # ~bool_masked_pos 表示可见的分块
        if bool_masked_pos is not None:
            batch_size, _, num_channels = embeddings.shape
            # 使用掩码过滤掉不可见的分块
            embeddings = embeddings[~bool_masked_pos]
            # 重塑张量形状，以重新整理分块维度
            embeddings = embeddings.reshape(batch_size, -1, num_channels)
    
        # 返回处理后的嵌入张量
        return embeddings
class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config):
        super().__init__()
        
        # 初始化VideoMAEPatchEmbeddings 类
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        # 确保image_size和patch_size为可迭代集合
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 设定image_size和patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        )
        # 设定num_channels和num_patches
        self.num_channels = num_channels
        self.num_patches = num_patches
        # 初始化卷积层projection
        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, pixel_values):
        # 获取输入pixel_values的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 检查通道维度是否正确
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 检查输入图像大小是否匹配模型参数设置
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}."
            )
        # 将维度重新排列为(batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        # 使用projection进行特征提取并展平
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class VideoMAESelfAttention(nn.Module):
    # 初始化方法，接受一个 VideoMAEConfig 类型的参数
    def __init__(self, config: VideoMAEConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏层大小不能被注意力头的数量整除，并且配置中没有嵌入大小的属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # 如果配置中指定了 qkv_bias 为 True，则创建查询和值的偏置参数
        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重塑为注意力得分所需的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状，将最后两个维度分别设置为注意力头的数量和注意力头的大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重塑输入张量的形状
        x = x.view(new_x_shape)
        # 对张量进行转置，交换最后两个维度的位置
        return x.permute(0, 2, 1, 3)

    # 前向传播方法
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
        # 定义函数参数和返回值的类型注释
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 创建一个与self.v_bias形状相同的零张量，并且不需要梯度
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        # 使用线性函数计算keys
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        # 使用线性函数计算values
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        # 使用线性函数计算queries
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)

        # 对keys进行维度转换
        key_layer = self.transpose_for_scores(keys)
        # 对values进行维度转换
        value_layer = self.transpose_for_scores(values)
        # 对queries进行维度转换
        query_layer = self.transpose_for_scores(queries)

        # 计算"query"和"key"的点积以获得原始的注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 通过除以注意力头大小的平方根来对注意力得分进行归一化
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力得分进行softmax处理，将其归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 进行dropout操作，随机将一些注意力概率设置为0
        attention_probs = self.dropout(attention_probs)

        # 如果输入了head_mask，则将注意力概率与head_mask相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，通过将注意力概率与value_layer的点积来实现
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对context_layer进行维度变换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出attention_probs，则返回context_layer和attention_probs，否则只返回context_layer
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制而来，将ViT换成VideoMAE
class VideoMAESelfOutput(nn.Module):
    """
    The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 使用线性层转换隐藏状态的维度
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 使用dropout进行正则化

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 使用线性层转换隐藏状态的维度
        hidden_states = self.dropout(hidden_states)  # 使用dropout对隐藏状态进行处理

        return hidden_states  # 返回处理后的隐藏状态


# 从transformers.models.vit.modeling_vit.ViTAttention复制而来，将ViT换成VideoMAE
class VideoMAEAttention(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.attention = VideoMAESelfAttention(config)  # 创建VideoMAESelfAttention对象
        self.output = VideoMAESelfOutput(config)  # 创建VideoMAESelfOutput对象
        self.pruned_heads = set()  # 创建一个空集合用于存储被删减的头部

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 删减线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被删减的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)  # 使用attention模块处理隐藏状态

        attention_output = self.output(self_outputs[0], hidden_states)  # 使用output模块处理attention输出和隐藏状态

        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要输出attention，将其加入到outputs中
        return outputs  # 返回输出


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制而来，将ViT换成VideoMAE
class VideoMAEIntermediate(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)  # 使用线性层进行维度转换
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]  # 根据配置选择合适的激活函数
        else:
            self.intermediate_act_fn = config.hidden_act  # 使用配置中指定的激活函数
    # 定义一个前向传播函数，接受隐藏状态张量并返回张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态并覆盖原隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理隐藏状态并覆盖原隐藏状态张量
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput复制，并将ViT更改为VideoMAE
class VideoMAEOutput(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层将输入状态转换为hidden_size维度
        hidden_states = self.dense(hidden_states)
        # 对hidden_states进行dropout
        hidden_states = self.dropout(hidden_states)

        # 将hidden_states与输入张量相加
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTLayer复制，并将ViT更改为VideoMAE
class VideoMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 配置块的前馈向前分配的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 创建一个VideoMAEAttention对象
        self.attention = VideoMAEAttention(config)
        # 创建一个VideoMAEIntermediate对象
        self.intermediate = VideoMAEIntermediate(config)
        # 创建一个VideoMAEOutput对象
        self.output = VideoMAEOutput(config)
        # 创建一个LayerNorm对象，在输入层之前进行归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个LayerNorm对象，在输入层之后进行归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用layernorm_before对隐藏状态进行归一化
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在VideoMAE中，进行了layernorm层
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取self_attention_outputs中的注意力输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，则在outputs中添加自注意力
        outputs = self_attention_outputs[1:]  

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在这里也应用了layer_norm前馈的均值
        layer_output = self.layernorm_after(hidden_states)
        # 通过intermediate层处理layer_output
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        # 在outputs中添加层输出
        outputs = (layer_output,) + outputs

        return outputs


# 从transformers.models.vit.modeling_vit.ViTEncoder复制，并将ViT更改为VideoMAE
class VideoMAEEncoder(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.config = config
        # 创建一个含有config.num_hidden_layers个VideoMAELayer对象的ModuleList
        self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 定义模型的前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化存储所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力分布，则初始化存储所有注意力分布的元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个层进行前向传播
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，若未提供则置为None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用渐变检查点且处于训练状态，则使用渐变检查点函数执行当前层的前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播方法
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力分布，则将当前层的注意力分布添加到元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典结果，则返回包含有效值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回具有各项内容的BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class VideoMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 指定配置类
    config_class = VideoMAEConfig
    # 模型前缀
    base_model_prefix = "videomae"
    # 主输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或3D卷积层
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # 根据配置中的初始化范围，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)


VIDEOMAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VideoMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIDEOMAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEModel(VideoMAEPreTrainedModel):
    # 初始化模型，设置配置
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置保存在类中
        self.config = config

        # 创建视频MAE嵌入层和编码器
        self.embeddings = VideoMAEEmbeddings(config)
        self.encoder = VideoMAEEncoder(config)

        # 根据配置选择是否使用均值池化，初始化LayerNorm（如果不使用均值池化）
        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型的attention头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接收像素值、遮挡位置、头部遮挡等参数
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class VideoMAEDecoder(nn.Module):
    # 定义视频 MAE 解码器类，继承自 PyTorch 的 nn.Module 类
    def __init__(self, config, num_patches):
        # 初始化函数，接收配置和补丁数作为参数
        super().__init__()
        # 调用父类的初始化函数

        # 计算解码器的标签数
        decoder_num_labels = config.num_channels * config.tubelet_size * config.patch_size**2

        # 深拷贝配置
        decoder_config = deepcopy(config)
        # 设置解码器的隐藏大小为配置中的解码器隐藏大小
        decoder_config.hidden_size = config.decoder_hidden_size
        # 设置解码器的隐藏层数为配置中的解码器隐藏层数
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        # 设置解码器的注意力头数为配置中的解码器注意力头数
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        # 设置解码器的中间层大小为配置中的解码器中间层大小
        decoder_config.intermediate_size = config.decoder_intermediate_size
        # 创建解码器层列表，包含指定数量的 VideoMAELayer 实例
        self.decoder_layers = nn.ModuleList(
            [VideoMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        # 实例化 LayerNorm 层，用于规范化隐藏状态
        self.norm = nn.LayerNorm(config.decoder_hidden_size)
        # 如果解码器标签数大于 0，则使用线性层；否则使用恒等映射
        self.head = (
            nn.Linear(config.decoder_hidden_size, decoder_num_labels) if decoder_num_labels > 0 else nn.Identity()
        )

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
        # 保存配置
        self.config = config

    def forward(
        self,
        hidden_states,
        return_token_num,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 前向传播函数，接收隐藏状态、返回 token 数、是否输出注意力、是否输出隐藏状态、是否返回字典作为参数

        # 如果输出隐藏状态为真，则初始化空元组以保存所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力为真，则初始化空元组以保存所有注意力权重
        all_self_attentions = () if output_attentions else None
        # 遍历解码器的每个层
        for i, layer_module in enumerate(self.decoder_layers):
            # 如果输出隐藏状态为真，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果梯度检查点为真且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力为真，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为真，则将当前隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果返回 token 数大于 0，则仅保留最后返回的 token 数量
        if return_token_num > 0:
            hidden_states = hidden_states[:, -return_token_num:]

        # 对隐藏状态进行规范化
        hidden_states = self.norm(hidden_states)
        # 将隐藏状态投影到预测器
        logits = self.head(hidden_states)

        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 VideoMAEDecoderOutput 对象
        return VideoMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)


@add_start_docstrings(
    "The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForPreTraining(VideoMAEPreTrainedModel):
    # 用于自监督预训练的 VideoMAE 模型，其顶部是解码器
    # 初始化函数，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置参数保存到实例变量中
        self.config = config

        # 创建 VideoMAEModel 实例
        self.videomae = VideoMAEModel(config)

        # 创建线性层，用于编码器到解码器的映射
        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        # 创建用于掩码标记的参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        # 创建位置编码表
        self.position_embeddings = get_sinusoid_encoding_table(
            self.videomae.embeddings.num_patches, config.decoder_hidden_size
        )

        # 创建 VideoMAEDecoder 实例
        self.decoder = VideoMAEDecoder(config, num_patches=self.videomae.embeddings.num_patches)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受像素值、掩码位置、头掩码、输出注意力、输出隐藏状态、返回字典等参数
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用 VideoMAE 模型进行视频分类的模型转换器，顶部有一个视频分类头（线性层位于所有标记的平均池化隐藏状态之上），例如用于 ImageNet
@add_start_docstrings(
    """VideoMAE Model transformer with a video classification head on top (a linear layer on top of the average pooled hidden
    states of all tokens) e.g. for ImageNet.""",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForVideoClassification(VideoMAEPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 VideoMAE 模型
        self.videomae = VideoMAEModel(config)

        # 分类器头部
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```