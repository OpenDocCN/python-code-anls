# `.\models\vit\modeling_vit.py`

```
# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViT model."""

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vit import ViTConfig

# Get logger for this module
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# List of pretrained ViT model archives
VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-base-patch16-224",
    # See all ViT models at https://huggingface.co/models?filter=vit
]


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # Initialize CLS token as a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Initialize mask token if `use_mask_token` is True
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None

        # Initialize patch embeddings
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        # Initialize position embeddings for patches and CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        
        # Dropout layer with dropout probability specified in config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Store configuration
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取嵌入张量中的补丁数量
        num_patches = embeddings.shape[1] - 1
        # 获取预训练位置编码张量中的位置数量
        num_positions = self.position_embeddings.shape[1] - 1
        # 如果补丁数量和位置数量相等，并且输入高度和宽度相等，则直接返回位置编码张量
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 从位置编码张量中获取类别位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        # 从位置编码张量中获取补丁位置编码
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 获取嵌入张量的最后一个维度（表示特征维度）
        dim = embeddings.shape[-1]
        # 计算调整后的高度和宽度
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # 添加一个小数以避免插值时的浮点数误差
        h0, w0 = h0 + 0.1, w0 + 0.1
        # 重塑补丁位置编码张量的形状
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        # 将补丁位置编码张量的维度顺序重新排列
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        # 使用双三次插值对补丁位置编码张量进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        # 断言调整后的高度和宽度与插值后的张量形状一致
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        # 将补丁位置编码张量的维度顺序重新排列
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # 在类别位置编码张量和调整后的补丁位置编码张量之间进行拼接，并返回结果张量
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    ) -> torch.Tensor:
        # 获取输入张量的维度信息：批大小、通道数、高度、宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 使用自定义函数将像素值转换为补丁嵌入
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果存在布尔掩码，处理被掩码的位置
        if bool_masked_pos is not None:
            # 获取嵌入的序列长度
            seq_length = embeddings.shape[1]
            # 扩展掩码令牌以匹配批大小和序列长度
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建掩码，将被掩码的可视令牌替换为掩码令牌
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将[CLS]令牌添加到嵌入的补丁令牌中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码到每个令牌
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入进行丢弃操作，以防止过拟合
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小不是可迭代对象，转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 如果patch大小不是可迭代对象，转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层进行投影，将输入的通道数转换为隐藏层大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入通道数是否与配置中的通道数匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        # 如果不插值位置编码，检查输入图像大小是否与配置中的图像大小匹配
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 对输入的像素值应用投影，并将结果展平并转置以生成嵌入向量
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，同时检查是否有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值线性层，带有可选的偏置项
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化dropout层，用于注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 对输入张量 x 进行形状转换，将最后两维重新组合成指定的形状，
    # 前面维度保持不变，以便后续计算注意力
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

# 定义前向传播函数，接受隐藏状态、头部掩码和是否输出注意力概率等参数
def forward(
    self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # 计算混合的查询向量
    mixed_query_layer = self.query(hidden_states)

    # 对键（key）和值（value）进行形状转换，以便计算注意力分数
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    # 计算注意力分数，即查询向量与键向量的点积
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    # 将注意力分数除以 sqrt(注意力头的大小)，以减少梯度消失问题
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # 对注意力分数进行 softmax 操作，得到注意力概率
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # 使用 dropout 对注意力概率进行随机置零，以防止过拟合
    attention_probs = self.dropout(attention_probs)

    # 如果存在头部掩码，则将注意力概率与掩码相乘
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # 计算加权后的值向量，即注意力概率与值向量的矩阵乘积
    context_layer = torch.matmul(attention_probs, value_layer)

    # 对结果进行形状转换，将头部维度重新合并为最后两个维度
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    # 根据是否输出注意力概率，选择输出的结果
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 Dropout 层，用于随机置零输入张量中的一些元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的输出使用 Dropout 进行随机置零
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 创建一个自注意力模块，用于计算注意力分布
        self.attention = ViTSelfAttention(config)
        # 创建一个自定义的输出模块，用于处理自注意力模块的输出
        self.output = ViTSelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的注意力头信息
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 调用剪枝函数，找到需要剪枝的注意力头并返回
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 对注意力模块的查询、键、值以及输出层的全连接层进行剪枝
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的注意力头信息
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用自注意力模块进行前向传播
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将自注意力模块的输出传递给输出模块进行处理
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将它们添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力权重，将它们添加到输出元组中
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入维度转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用配置中选择的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    # 在 ViTOutput 类中的代码将在下一个问题中继续进行。
    # 初始化函数，用于初始化类的实例
    def __init__(self, config: ViTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为配置中的中间大小，输出大小为配置中的隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个dropout层，用于在训练过程中随机置零输入张量的部分元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将dropout后的隐藏状态与输入张量相加
        hidden_states = hidden_states + input_tensor

        # 返回前向传播的结果张量
        return hidden_states
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 设置块大小以进行前向传播分块处理
        self.seq_len_dim = 1  # 序列长度维度设定为1，通常用于处理输入序列的长度
        self.attention = ViTAttention(config)  # 初始化注意力机制模块
        self.intermediate = ViTIntermediate(config)  # 初始化中间层模块
        self.output = ViTOutput(config)  # 初始化输出层模块
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化前向传播前的层归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化前向传播后的层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在ViT中，先应用层归一化再进行自注意力计算
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力的输出
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，将其添加到输出中

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViT中，自注意力后也应用层归一化
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)  # 中间层处理

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)  # 输出层处理

        outputs = (layer_output,) + outputs  # 将处理后的输出添加到结果中

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])  # 创建多层ViTLayer组成的层列表
        self.gradient_checkpointing = False  # 梯度检查点设为False，通常用于优化内存消耗

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ) -> Union[tuple, BaseModelOutput]:
        # 如果不需要输出隐藏状态，则初始化为空元组；否则设为None，以便后续添加隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则设为None，以便后续添加注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且在训练阶段
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 正常情况下，调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出
        if not return_dict:
            # 返回非空元素的元组，包括 hidden_states, all_hidden_states, all_self_attentions
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回 BaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    """
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    # 接受输入参数：
    # pixel_values: 表示像素值的张量，形状为 `(batch_size, num_channels, height, width)`
    #               可以使用 `AutoImageProcessor` 获取。详见 `ViTImageProcessor.__call__` 的说明。
    # head_mask: 可选参数，形状可以是 `(num_heads,)` 或者 `(num_layers, num_heads)` 的张量。
    #            用于掩盖自注意力模块中选定的头部。掩盖值在 `[0, 1]` 范围内：
    #            - 1 表示该头部**未被掩盖**，
    #            - 0 表示该头部**被掩盖**。
    # output_attentions: 可选参数，布尔值，是否返回所有注意力层的注意力张量。
    #                    返回的张量中的 `attentions` 字段包含更多细节。
    # output_hidden_states: 可选参数，布尔值，是否返回所有层的隐藏状态。
    #                       返回的张量中的 `hidden_states` 字段包含更多细节。
    # interpolate_pos_encoding: 可选参数，布尔值，是否插值预训练的位置编码。
    # return_dict: 可选参数，布尔值，是否返回一个 `~utils.ModelOutput` 对象而不是普通元组。
"""
@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
"""



class ViTModel(ViTPreTrainedModel):
    """
    ViT Model class inheriting from ViTPreTrainedModel.
    """

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        """
        Initializes a ViTModel instance.

        Args:
            config (ViTConfig): Configuration class instance defining model architecture.
            add_pooling_layer (bool): Whether to add a pooling layer on top of the encoder.
            use_mask_token (bool): Whether to use a mask token for the model.

        """
        super().__init__(config)
        self.config = config

        # Initialize embeddings and encoder
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        # Layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Optional pooling layer
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        """
        Returns the patch embeddings used as input to the model.
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (Dict[int, List[int]]): Dictionary of layers and heads to prune.

        See base class PreTrainedModel for more details.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    """
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    """



    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 根据输入或者配置决定是否输出注意力矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据输入或者配置决定是否输出隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据输入或者配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # head_mask 中的 1.0 表示保留对应的注意力头
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或者 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: 可能有更干净的方式将输入转换（从 `ImageProcessor` 方面考虑？）
        # 如果像素值的数据类型与期望的数据类型不匹配，则转换像素值的数据类型
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # 获取嵌入输出，根据输入的布尔掩码位置和插值位置编码进行插值
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # 编码器处理，传递嵌入输出，根据需要传递头部掩码、是否输出注意力、是否输出隐藏层状态、是否使用返回字典
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不使用返回字典，返回头部输出和编码器输出的其余部分
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 使用自定义的返回类返回模型的输出，包括最终的隐藏状态、池化输出、隐藏层状态和注意力矩阵
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        # 定义一个全连接层，输入和输出的大小都是隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数，使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 通过取第一个标记对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数到全连接层输出
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForMaskedImageModeling(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        # 初始化ViT模型，设置不添加池化层和使用掩码标记
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        # 定义解码器
        self.decoder = nn.Sequential(
            # 定义一个2D卷积层，输入通道数为隐藏层大小，输出通道数为config中的计算结果
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            # 像素重排操作
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化函数，用于初始化一个视觉Transformer模型
    def __init__(self, config: ViTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建一个不带池化层的ViT模型实例
        self.vit = ViTModel(config, add_pooling_layer=False)

        # 分类器头部
        # 如果标签数量大于0，则创建一个线性层作为分类器；否则创建一个恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 前向传播函数，接收像素值、头部掩码、标签等参数，并返回模型的输出
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入的像素值和其他参数传递给 Vision Transformer 模型进行处理
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # 从模型输出中提取序列输出（Sequence Output）
        sequence_output = outputs[0]

        # 将序列输出的首个位置的特征向量输入分类器，得到分类器的 logits
        logits = self.classifier(sequence_output[:, 0, :])

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以实现模型的并行计算
            labels = labels.to(logits.device)
            # 根据问题类型动态确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回一个包含 logits 和额外输出的元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个 ImageClassifierOutput 对象
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```