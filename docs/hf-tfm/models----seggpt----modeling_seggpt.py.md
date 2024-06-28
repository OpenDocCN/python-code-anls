# `.\models\seggpt\modeling_seggpt.py`

```
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SegGpt model."""


import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

# Importing utilities and components from the HuggingFace library
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# Import configuration specific to SegGpt
from .configuration_seggpt import SegGptConfig


# Get the logger for this module
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SegGptConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "BAAI/seggpt-vit-large"
_EXPECTED_OUTPUT_SHAPE = [3, 896, 448]

# List of pretrained model archive names specific to SegGpt
SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BAAI/seggpt-vit-large",
    # See all SegGpt models at https://huggingface.co/models?filter=seggpt
]


@dataclass
class SegGptEncoderOutput(ModelOutput):
    """
    Output type of [`SegGptEncoderOutput`].
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of *torch.FloatTensor* (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
        intermediate_hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.intermediate_hidden_state_indices` is set):
            Tuple of `torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`.
            Each element in the Tuple corresponds to the output of the layer specified in `config.intermediate_hidden_state_indices`.
            Additionaly, each feature passes through a LayerNorm.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # intermediate_hidden_states 是一个可选的类型为 Tuple[torch.FloatTensor] 的变量，初始值为 None。
    intermediate_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class SegGptImageSegmentationOutput(ModelOutput):
    """
    Output type of [`SegGptImageSegmentationOutput`].

    Args:
        loss (`torch.FloatTensor`, `optional`, returned when `labels` is provided):
            The loss value.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted masks.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
    """

    loss: Optional[torch.FloatTensor] = None  # 可选的损失值
    pred_masks: Optional[torch.FloatTensor] = None  # 可选的预测掩码
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 可选的隐藏状态元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 可选的注意力元组


# Copied from transformers.models.sam.modeling_sam.SamPatchEmbeddings with Sam->SegGpt
class SegGptPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size  # 图像尺寸
        self.patch_size = patch_size  # 补丁尺寸
        self.num_channels = num_channels  # 通道数
        self.num_patches = num_patches  # 补丁数量

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)  # 投影像素值到补丁嵌入的张量维度
        return embeddings


class SegGptEmbeddings(nn.Module):
    """
    Placeholder for SegGptEmbeddings class definition.
    """
    Construct the embeddings from patch, position embeddings for input and prompt.
    """

    # 定义一个名为SegGptEmbeddings的类，继承自父类nn.Module
    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()

        # 定义用于掩码的张量参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # 定义输入分段标记的张量参数
        self.segment_token_input = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # 定义提示分段标记的张量参数
        self.segment_token_prompt = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # 定义语义类型标记的张量参数
        # token for seg types
        self.type_token_semantic = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # 定义实例类型标记的张量参数
        self.type_token_instance = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))

        # 初始化图像块嵌入对象
        self.patch_embeddings = SegGptPatchEmbeddings(config)

        # 计算位置嵌入的数量
        num_positions = (config.pretrain_image_size // config.patch_size) ** 2 + 1
        # 定义位置嵌入的张量参数
        self.position_embeddings = nn.Parameter(torch.randn(1, num_positions, config.hidden_size))
        # 定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 定义一个插值位置编码的方法
    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        # 获取位置编码中的图像块位置嵌入
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 计算图像块的数量
        num_patches = patch_pos_embed.shape[1]
        # 计算预训练图像块大小的平方根
        pretrain_patch_size = int(math.sqrt(num_patches))

        # 如果预训练图像块大小与给定的高度或宽度不匹配，则进行插值处理
        if pretrain_patch_size != height or pretrain_patch_size != width:
            # 使用双三次插值方法对位置编码进行插值
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, pretrain_patch_size, pretrain_patch_size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )

            # 将插值后的位置编码张量进行维度调整，并返回
            return patch_pos_embed.permute(0, 2, 3, 1)
        else:
            # 如果不需要插值，则直接返回原始的位置编码张量
            return patch_pos_embed.reshape(1, height, width, -1)

    # 定义前向传播方法
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        embedding_type: Optional[str] = None,
        # 继续定义其他参数
    ) -> torch.Tensor:
        # 使用self.patch_embeddings方法将像素值转换为输入嵌入
        input_embeddings = self.patch_embeddings(pixel_values)
        # 使用self.patch_embeddings方法将提示像素值转换为提示嵌入
        prompt_embeddings = self.patch_embeddings(prompt_pixel_values)

        # 获取输入嵌入的维度信息
        batch_size, patch_height, patch_width, _ = input_embeddings.shape

        # 扩展mask_token以匹配输入嵌入的形状
        mask_token = self.mask_token.expand(batch_size, patch_height, patch_width, -1)
        # 使用bool_masked_pos创建一个掩码，将掩码处的视觉标记替换为mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, patch_height, patch_width, 1)
        prompt_embeddings = prompt_embeddings * (1 - w) + mask_token * w

        # 如果未指定embedding_type，则默认为"instance"
        embedding_type = embedding_type if embedding_type is not None else "instance"

        # 添加位置编码到每个标记
        pos_embed = self.interpolate_pos_encoding(patch_height, patch_width)

        # 添加段标记到输入嵌入和提示嵌入
        input_embeddings = input_embeddings + self.segment_token_input
        prompt_embeddings = prompt_embeddings + self.segment_token_prompt

        # 跳过CLS后，添加位置编码到输入嵌入和提示嵌入
        input_embeddings = input_embeddings + pos_embed
        prompt_embeddings = prompt_embeddings + pos_embed

        # 根据embedding_type选择对应的类型嵌入
        if embedding_type == "semantic":
            type_embedding = self.type_token_semantic
        elif embedding_type == "instance":
            type_embedding = self.type_token_instance
        else:
            raise ValueError(f"Embedding type should be either 'semantic' or 'instance', but got {embedding_type}")

        # 添加类型嵌入到输入嵌入和提示嵌入
        input_embeddings = input_embeddings + type_embedding
        prompt_embeddings = prompt_embeddings + type_embedding

        # 将输入嵌入和提示嵌入连接起来形成最终的嵌入张量
        embeddings = torch.cat((input_embeddings, prompt_embeddings), dim=0)

        # 返回最终的嵌入张量
        return embeddings
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Add decomposed relative positional embeddings to attention scores.

        Args:
            attn (torch.Tensor):
                Attention scores.
            query (torch.Tensor):
                Query tensor.
            rel_pos_h (torch.Tensor):
                Relative positional embeddings along height dimension.
            rel_pos_w (torch.Tensor):
                Relative positional embeddings along width dimension.
            q_size (Tuple[int, int]):
                Size of the query tensor.
            k_size (Tuple[int, int]):
                Size of the key tensor.

        Returns:
            Updated attention scores with added decomposed relative positional embeddings.
        """

        # Get relative position embeddings based on query and key sizes
        rel_pos_h = self.get_rel_pos(q_size[0], k_size[0], rel_pos_h)
        rel_pos_w = self.get_rel_pos(q_size[1], k_size[1], rel_pos_w)

        # Add relative position embeddings to attention scores
        attn += torch.matmul(query, rel_pos_h.unsqueeze(0)) + torch.matmul(query, rel_pos_w.unsqueeze(0).transpose(-2, -1))
        
        return attn
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        # 解构 q_size 元组，获取查询张量的高度和宽度
        query_height, query_width = q_size
        # 解构 k_size 元组，获取键张量的高度和宽度
        key_height, key_width = k_size
        
        # 获取高度轴的相对位置编码，形状为 (batch_size, query_height, query_width, key_height, channel)
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        # 获取宽度轴的相对位置编码，形状为 (batch_size, query_height, query_width, key_width, channel)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        # 获取查询张量的批量大小、高度、宽度和维度
        batch_size, _, dim = query.shape
        # 将查询张量重塑为四维张量 (batch_size, query_height, query_width, dim)
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        
        # 计算高度轴的相对位置编码与查询张量的乘积，形状为 (batch_size, query_height, query_width, key_height)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        # 计算宽度轴的相对位置编码与查询张量的乘积，形状为 (batch_size, query_height, query_width, key_width)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        
        # 将注意力图重塑为五维张量 (batch_size, query_height, query_width, key_height, key_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        # 将注意力图与高度轴和宽度轴的相对位置编码相加
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        
        # 将注意力图重塑为二维张量 (batch_size, query_height * query_width, key_height * key_width)
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        # 返回添加了相对位置编码的注意力图
        return attn
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        # 获取隐藏状态的形状信息，batch_size为批大小，height为高度，width为宽度，_为通道数
        batch_size, height, width, _ = hidden_states.shape
        
        # 使用self.qkv对隐藏状态进行qkv计算，结果形状为(3, batch_size, num_attention_heads, height * width, embed_dim)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)  # 重塑形状以便后续操作
            .permute(2, 0, 3, 1, 4)  # 转置以便得到q, k, v分量
        )
        
        # 将qkv分解为query, key, value三个部分，形状为(batch_size * num_attention_heads, height * width, embed_dim)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        
        # 计算注意力权重，形状为(batch_size * num_attention_heads, height * width, height * width)
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)
        
        # 如果使用相对位置编码，则对注意力权重进行处理
        if self.use_relative_position_embeddings:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )
        
        # 对注意力权重进行softmax操作，保留query的数据类型
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        
        # 如果需要输出注意力权重，则进行特定的形状重塑操作，否则attn_weights_reshaped为None
        if output_attentions:
            # 这个操作有些笨拙，但是需要确保attn_weights保持其梯度。
            # 为了做到这一点，attn_weights必须进行两次重塑，并且在接下来的使用中需要重用它们
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_attention_heads, height * width, -1)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_attention_heads, height * width, -1)
        else:
            attn_weights_reshaped = None
        
        # 计算注意力输出，形状为(batch_size, num_attention_heads, height, width, embed_dim)
        attn_output = (attn_weights @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        # 调整输出的形状，使其变为(batch_size, height, width, num_attention_heads * embed_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        
        # 对注意力输出进行投影，形状为(batch_size, height, width, embed_dim)
        attn_output = self.proj(attn_output)
        
        # 返回注意力输出和注意力权重的重塑形状（如果需要）
        return (attn_output, attn_weights_reshaped)
# 从transformers.models.sam.modeling_sam.SamMLPBlock复制到SegGptMlp
class SegGptMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度是config.hidden_size，输出维度是config.mlp_dim
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        # 创建另一个线性层，输入维度是config.mlp_dim，输出维度是config.hidden_size
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        # 选择激活函数，根据config.hidden_act从预定义的ACT2FN字典中选择对应的函数
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入的hidden_states应用第一个线性层
        hidden_states = self.lin1(hidden_states)
        # 应用选择的激活函数
        hidden_states = self.act(hidden_states)
        # 对应用激活函数后的结果应用第二个线性层
        hidden_states = self.lin2(hidden_states)
        # 返回处理后的hidden_states作为输出
        return hidden_states


# 从transformers.models.beit.modeling_beit.drop_path复制
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本丢弃路径（随机深度），应用于残差块的主路径中。

    Ross Wightman的注释：这与我为EfficientNet等网络创建的DropConnect实现相同，但原始名称误导，因为'Drop Connect'是另一篇论文中的一种不同的丢弃形式...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择更改层和参数名称为'drop path'，而不是将DropConnect作为层名称并使用'survival rate'作为参数。
    """
    if drop_prob == 0.0 or not training:
        # 如果drop_prob为0或者不处于训练模式，则直接返回输入
        return input
    keep_prob = 1 - drop_prob
    # 创建一个与输入张量形状相同的随机张量，值在[keep_prob, 1.0)之间
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 应用丢弃路径操作，将输入张量按照keep_prob进行缩放
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath复制到SegGptDropPath
class SegGptDropPath(nn.Module):
    """按样本丢弃路径（随机深度），应用于残差块的主路径中。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数，传入hidden_states、drop_prob和当前模块是否处于训练模式
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegGptLayer(nn.Module):
    def __init__(self, config: SegGptConfig, drop_path_rate: float) -> None:
        super().__init__()
        # 创建一个SegGptAttention对象，使用给定的config
        self.attention = SegGptAttention(config)
        # 创建一个SegGptMlp对象，使用给定的config
        self.mlp = SegGptMlp(config)
        # 如果drop_path_rate大于0.0，则创建一个SegGptDropPath对象，否则创建一个恒等映射(nn.Identity())
        self.drop_path = SegGptDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 创建一个LayerNorm层，输入维度是config.hidden_size，epsilon值是config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    # 定义神经网络的前向传播方法，接收多个输入参数并返回一个或两个张量的元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        ensemble_cond: int,
        feature_ensemble: bool = False,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        # 使用 self.attention 方法进行自注意力计算，先对 hidden_states 进行 layernorm 处理
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 SegGpt 中，在进行自注意力计算前先应用 layernorm
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]  # 提取自注意力计算的输出
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则将其添加到 outputs 中

        # 如果 feature_ensemble 为 True，且满足 ensemble_cond 条件
        if feature_ensemble and attention_output.shape[0] // 2 >= ensemble_cond:
            # 将 attention_output 拆分为 prompt 和 inputs
            prompt, inputs = attention_output.split(attention_output.shape[1] // 2, dim=1)
            # 如果 ensemble_cond 等于 2
            if ensemble_cond == 2:
                num_prompts = attention_output.shape[0] // 2
                # 对 inputs 进行形状调整和均值计算
                inputs = inputs.reshape(2, num_prompts, -1)
                inputs = inputs.mean(dim=1, keepdim=True).expand_as(inputs)
                inputs = inputs.reshape(*prompt.shape)
            else:
                # 对 inputs 进行均值计算和扩展
                inputs = inputs.mean(dim=0, keepdim=True).expand_as(inputs)
            # 拼接处理后的 prompt 和 inputs，并更新 attention_output
            attention_output = torch.cat([prompt, inputs], dim=1)

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states
        residual = hidden_states  # 保存残差连接后的 hidden_states

        # 在 self.layernorm_after 后应用 layernorm
        hidden_states = self.layernorm_after(hidden_states)
        # 通过 MLP 网络进行非线性变换
        hidden_states = self.mlp(hidden_states)
        # 第二个残差连接
        hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,) + outputs  # 更新 outputs，添加最终的 hidden_states

        return outputs  # 返回前向传播的结果
class SegGptEncoder(nn.Module):
    # SegGpt 编码器类，继承自 nn.Module
    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()
        self.config = config
        # 生成一个从0到配置的 drop_path_rate 的线性序列，并转换为 Python 列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 创建包含多个 SegGptLayer 实例的 ModuleList，每个实例使用不同的 drop_path_rate
        self.layers = nn.ModuleList([SegGptLayer(config, dpr[i]) for i in range(config.num_hidden_layers)])
        # 创建 LayerNorm 层，用于规范化隐藏状态的尺寸，设置 epsilon 为 config.layer_norm_eps
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否开启梯度检查点功能，默认为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_ensemble: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, SegGptEncoderOutput]:
        # 如果输出隐藏状态，则初始化一个空元组来存储所有的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组来存储所有的注意力权重
        all_self_attentions = () if output_attentions else None
        # 用于存储中间隐藏状态的列表
        intermediate_hidden_states = []

        # 遍历所有层
        for i, layer_module in enumerate(self.layers):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 根据当前层的索引判断是否需要多个提示来进行集成
            ensemble_cond = 2 if self.config.merge_index > i else 1

            # 如果开启梯度检查点功能并且正在训练，则使用梯度检查点函数来执行当前层的调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    ensemble_cond,
                    feature_ensemble,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播方法
                layer_outputs = layer_module(hidden_states, ensemble_cond, feature_ensemble, output_attentions)

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果当前层的索引等于配置的 merge_index，则执行合并操作
            if i == self.config.merge_index:
                hidden_states = (
                    hidden_states[: hidden_states.shape[0] // 2] + hidden_states[hidden_states.shape[0] // 2 :]
                ) * 0.5

            # 如果当前层的索引在配置的 intermediate_hidden_state_indices 中，则将规范化后的隐藏状态添加到中间隐藏状态列表中
            if i in self.config.intermediate_hidden_state_indices:
                intermediate_hidden_states.append(self.layernorm(hidden_states))

            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回一个元组，其中包含所有非空的结果项
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, intermediate_hidden_states]
                if v is not None
            )
        # 否则返回 SegGptEncoderOutput 对象，包含最后的隐藏状态、所有隐藏状态、所有注意力权重和中间隐藏状态列表
        return SegGptEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            intermediate_hidden_states=intermediate_hidden_states,
        )


# 从 transformers.models.convnext.modeling_convnext.ConvNextLayerNorm 复制并修改为 SegGptLayerNorm
class SegGptLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    # 定义一个支持两种数据格式（channels_last 或 channels_first）的 LayerNorm 类
    class LayerNorm(nn.Module):
        
        # 初始化方法，接受 normalized_shape（标准化的维度大小）、eps（防止除零的小常数，默认为 1e-6）、data_format（数据格式，默认为 channels_last）
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            # 初始化权重参数为 1，并将其包装为 nn.Parameter，使其可以被优化
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            # 初始化偏置参数为 0，并将其包装为 nn.Parameter，使其可以被优化
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            # 设置 eps 参数
            self.eps = eps
            # 检查数据格式是否为支持的 channels_last 或 channels_first
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError(f"Unsupported data format: {self.data_format}")
            # 存储标准化的维度信息
            self.normalized_shape = (normalized_shape,)
        
        # 前向传播方法，接受输入张量 x，返回标准化后的张量
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 如果数据格式是 channels_last，则使用 torch.nn.functional.layer_norm 函数进行标准化
            if self.data_format == "channels_last":
                x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            # 如果数据格式是 channels_first，则手动实现标准化过程
            elif self.data_format == "channels_first":
                # 保存输入张量的数据类型
                input_dtype = x.dtype
                # 将输入张量转换为 float 类型
                x = x.float()
                # 计算均值 u
                u = x.mean(1, keepdim=True)
                # 计算方差 s
                s = (x - u).pow(2).mean(1, keepdim=True)
                # 标准化过程：(x - u) / sqrt(s + eps)
                x = (x - u) / torch.sqrt(s + self.eps)
                # 将输出张量的数据类型转换回输入的数据类型
                x = x.to(dtype=input_dtype)
                # 应用权重和偏置调整
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            # 返回标准化后的张量
            return x
# 定义一个名为 SegGptDecoderHead 的类，继承自 nn.Module
class SegGptDecoderHead(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个 2D 卷积层，输入和输出通道数都是 config.decoder_hidden_size，卷积核大小为 3x3，填充为 1
        self.conv = nn.Conv2d(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
        )
        # 初始化一个 SegGptLayerNorm 实例，对输入进行归一化，通道顺序为 "channels_first"
        self.layernorm = SegGptLayerNorm(
            normalized_shape=config.decoder_hidden_size, eps=config.layer_norm_eps, data_format="channels_first"
        )
        # 根据配置选择激活函数，ACT2FN 是一个预定义的激活函数字典
        self.act_fct = ACT2FN[config.hidden_act]
        # 定义一个 1x1 的 2D 卷积层，将隐藏状态映射到 3 个通道，带有偏置
        self.head = nn.Conv2d(config.decoder_hidden_size, 3, kernel_size=1, bias=True)  # decoder to patch

    # 前向传播方法，接收输入 hidden_states
    def forward(self, hidden_states: torch.FloatTensor):
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的结果进行归一化
        hidden_states = self.layernorm(hidden_states)
        # 应用预定义的激活函数
        hidden_states = self.act_fct(hidden_states)
        # 将激活后的结果再次经过一个 1x1 卷积层
        hidden_states = self.head(hidden_states)

        return hidden_states


# 定义一个名为 SegGptDecoder 的类，继承自 nn.Module
class SegGptDecoder(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个线性层，用于将输入维度转换为 config.patch_size^2 * config.decoder_hidden_size 的输出维度
        self.decoder_embed = nn.Linear(
            config.hidden_size * len(config.intermediate_hidden_state_indices),
            config.patch_size**2 * config.decoder_hidden_size,
            bias=True,
        )
        # 初始化一个 SegGptDecoderHead 的实例，作为解码器的预测头部
        self.decoder_pred = SegGptDecoderHead(config)
        # 记录 patch 的大小
        self.patch_size = config.patch_size
        # 记录解码器隐藏层的大小
        self.decoder_hidden_size = config.decoder_hidden_size
        # 记录配置对象
        self.config = config

    # 定义一个辅助方法，用于重塑隐藏状态的形状
    def _reshape_hidden_states(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 获取输入的张量形状信息
        batch_size, patch_height, patch_width, _ = hidden_states.shape
        # 将输入的张量重塑为新的形状
        hidden_states = hidden_states.reshape(
            batch_size, patch_height, patch_width, self.patch_size, self.patch_size, self.decoder_hidden_size
        )
        # 对重塑后的张量进行维度排列变换
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        # 再次重塑为指定形状
        hidden_states = hidden_states.reshape(
            shape=(batch_size, -1, patch_height * self.patch_size, patch_width * self.patch_size)
        )

        return hidden_states

    # 前向传播方法，接收输入 hidden_states
    def forward(self, hidden_states: torch.FloatTensor):
        # 将输入的隐藏状态先经过线性层进行维度转换
        hidden_states = self.decoder_embed(hidden_states)
        # 调用辅助方法重塑隐藏状态的形状
        hidden_states = self._reshape_hidden_states(hidden_states)
        # 将重塑后的隐藏状态传入解码器的预测头部进行处理
        hidden_states = self.decoder_pred(hidden_states)

        return hidden_states


# 定义一个名为 SegGptPreTrainedModel 的类，继承自 PreTrainedModel
class SegGptPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化、预训练模型下载和加载的简单接口。
    """

    # 类属性：配置类为 SegGptConfig
    config_class = SegGptConfig
    # 模型基础名称前缀为 "model"
    base_model_prefix = "model"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表
    _no_split_modules = ["SegGptEmbeddings", "SegGptLayer"]
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 从配置中获取初始化的标准差
        std = self.config.initializer_range

        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用截断正态分布初始化权重，先将权重转换为 float32 类型以避免在 half 精度下出现 `trunc_normal_cpu` 未实现的问题，然后再转回原始的 dtype
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=std).to(
                module.weight.dtype
            )
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()

        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为全 1
            module.weight.data.fill_(1.0)

        # 如果模块是 SegGptAttention 类型
        elif isinstance(module, SegGptAttention):
            # 使用截断正态分布初始化相对位置编码的水平方向数据
            module.rel_pos_h.data = nn.init.trunc_normal_(
                module.rel_pos_h.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.rel_pos_h.dtype)
            # 使用截断正态分布初始化相对位置编码的垂直方向数据
            module.rel_pos_w.data = nn.init.trunc_normal_(
                module.rel_pos_w.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.rel_pos_w.dtype)

        # 如果模块是 SegGptEmbeddings 类型
        elif isinstance(module, SegGptEmbeddings):
            # 使用截断正态分布初始化位置嵌入数据
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.position_embeddings.dtype)
            
            # 初始化其他特殊令牌的数据，使用正态分布初始化
            torch.nn.init.normal_(module.mask_token, std=std)
            torch.nn.init.normal_(module.segment_token_input, std=std)
            torch.nn.init.normal_(module.segment_token_prompt, std=std)
            torch.nn.init.normal_(module.type_token_semantic, std=std)
            torch.nn.init.normal_(module.type_token_instance, std=std)
"""
    This model is a PyTorch `torch.nn.Module` subclass designed for SegGpt model architecture. Use it
    like any regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

    Parameters:
        config (`SegGptConfig`): Model configuration class containing all model parameters.
            Initializing with a config file loads the configuration settings only, not the model weights.
            Use `PreTrainedModel.from_pretrained` to load weights associated with the model.
"""

"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Input pixel values. These are obtained using `AutoImageProcessor`. See `SegGptImageProcessor.__call__`
            for detailed information.

        prompt_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Prompt-specific pixel values. These are obtained using `AutoImageProcessor`. See `SegGptImageProcessor.__call__`
            for detailed information.

        prompt_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Mask applied to prompts. This is obtained using `AutoImageProcessor`. See `SegGptImageProcessor.__call__`
            for detailed information.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean tensor indicating masked positions (1 for masked, 0 for not masked).

        feature_ensemble (`bool`, *optional*):
            Indicates whether to use feature ensemble. If `True`, the model uses feature ensemble when multiple prompts
            are present. If `False`, it does not. Relevant for few-shot inference on an input image with more than one prompt.

        embedding_type (`str`, *optional*):
            Type of embedding used for prompts. Can be 'instance' or 'semantic'.

        output_attentions (`bool`, *optional*):
            Whether to return the attentions tensors of all attention layers. See `attentions` in returned tensors
            for more details.

        output_hidden_states (`bool`, *optional*):
            Whether to return the hidden states of all layers. See `hidden_states` in returned tensors for more details.

        return_dict (`bool`, *optional*):
            Whether to return a `utils.ModelOutput` instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare SegGpt Model transformer outputting raw hidden-states without any specific head on top.",
    SEGGPT_START_DOCSTRING,
)
class SegGptModel(SegGptPreTrainedModel):
    def __init__(self, config: SegGptConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = SegGptEmbeddings(config)
        self.encoder = SegGptEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()



        def get_input_embeddings(self) -> SegGptPatchEmbeddings:
            return self.embeddings.patch_embeddings



        def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
            """
            Prunes heads of the model.
            
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                # Access each layer of the encoder and prune specified heads in the attention mechanism
                self.encoder.layer[layer].attention.prune_heads(heads)



        @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=SegGptEncoderOutput, config_class=_CONFIG_FOR_DOC)
        def forward(
            self,
            pixel_values: torch.Tensor,
            prompt_pixel_values: torch.Tensor,
            prompt_masks: torch.Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            feature_ensemble: Optional[bool] = None,
            embedding_type: Optional[str] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,






            ):
                """
                Forward pass of the SegGptModel.
                
                pixel_values: torch.Tensor of shape (batch_size, num_patches, embed_dim)
                    Tensor containing pixel values.
                prompt_pixel_values: torch.Tensor of shape (batch_size, num_prompt_patches, embed_dim)
                    Tensor containing prompt pixel values.
                prompt_masks: torch.Tensor of shape (batch_size, num_patches)
                    Mask to ignore prompt tokens.
                bool_masked_pos: Optional[torch.BoolTensor], optional
                    Boolean mask for masked positions, by default None.
                feature_ensemble: Optional[bool], optional
                    Whether to use feature ensemble, by default None.
                embedding_type: Optional[str], optional
                    Type of embedding used, by default None.
                output_attentions: Optional[bool], optional
                    Whether to output attentions, by default None.
                output_hidden_states: Optional[bool], optional
                    Whether to output hidden states, by default None.
                return_dict: Optional[bool], optional
                    Whether to return a dictionary, by default None.






                Returns:
                    SegGptEncoderOutput or Tuple(torch.Tensor), torch.Tensor))
                    A SegGptEncoderOutput (if return_dict=True) or a tuple of torch.Tensors
                    (prompt_tokens, prompt_mask, prompt_patch_embedding)
                """
                # Forward pass logic would be implemented here, detailing how inputs are processed
                # through the layers of the model to produce the desired outputs.
# 定义一个函数，将输入的张量切分成指定大小的图块，并重新组织形状
def patchify(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    # 获取张量的批量大小、通道数、高度和宽度
    batch_size, num_channels, height, width = tensor.shape
    # 计算图块的高度和宽度
    patch_height = height // patch_size
    patch_width = width // patch_size

    # 将张量重新形状为(batch_size, num_channels, patch_height, patch_size, patch_width, patch_size)
    tensor = tensor.reshape(shape=(batch_size, num_channels, patch_height, patch_size, patch_width, patch_size))
    # 对张量进行维度置换，调整为(batch_size, patch_height, patch_width, patch_size, patch_size, num_channels)
    tensor = tensor.permute(0, 2, 4, 3, 5, 1)
    # 再次重新形状为(batch_size, patch_height * patch_width, patch_size^2 * num_channels)
    tensor = tensor.reshape(shape=(batch_size, patch_height * patch_width, patch_size**2 * 3))

    return tensor


# 定义一个函数，将输入的张量反转回原始的高度和宽度
def unpatchify(tensor: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    # 获取张量的批量大小
    batch_size = tensor.shape[0]
    # 推断出图块的大小
    patch_size = int((tensor.shape[-1] / 3) ** 0.5)
    # 检查图块数量是否与给定的patch_height和patch_width相匹配
    if patch_height * patch_width != tensor.shape[1]:
        raise ValueError(f"Number of patches {tensor.shape[1]} does not match patch height and width.")

    # 将张量重新形状为(batch_size, patch_height, patch_width, patch_size, patch_size, 3)
    tensor = tensor.reshape(shape=(batch_size, patch_height, patch_width, patch_size, patch_size, 3))
    # 对张量进行维度置换，调整为(batch_size, 3, patch_height * patch_size, patch_width * patch_size)
    tensor = tensor.permute(0, 5, 1, 3, 2, 4)

    return tensor


# 定义一个用于语义分割和GPT模型的损失函数类
class SegGptLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化损失函数的参数
        self.beta = config.beta
        self.patch_size = config.patch_size

    # 前向传播方法，计算损失
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        prompt_pixel_values: torch.FloatTensor,
        pred_masks: torch.FloatTensor,
        labels: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
        ```
        ):
        # 此处应该继续注释forward方法的其余部分，但这里不做展示
        pass  # 在此处插入pass语句
        """
        计算预测掩码与实际掩码之间的L1损失。

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                合并的像素值，来自提示图像和输入图像。

            prompt_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                来自掩码提示的合并像素值。

            pred_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                预测的掩码。

            labels (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                输入图像的实际掩码。

            bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
                布尔掩码位置。指示哪些补丁被掩盖（1），哪些没有（0）。

        Returns:
            `torch.FloatTensor`: 预测掩码与实际掩码之间的平均L1损失。
        """
        # 根据掩码位置创建掩码
        mask = bool_masked_pos[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        # 将掩码映射回原始尺寸
        mask = unpatchify(mask, pixel_values.shape[1] // self.patch_size, pixel_values.shape[2] // self.patch_size)
        # 将掩码提示中的虚拟掩码改为实际标签值
        prompt_pixel_values = prompt_pixel_values.clone()
        prompt_pixel_values[:, :, prompt_pixel_values.shape[2] // 2 :, :] = labels
        # 计算平滑L1损失，不进行缩减，并根据掩码应用损失
        loss = F.smooth_l1_loss(pred_masks, prompt_pixel_values, reduction="none", beta=self.beta)
        loss = (loss * mask).sum() / mask.sum()  # 计算移除补丁后的平均损失

        return loss
# 添加类的文档字符串，描述 SegGptForImageSegmentation 类的作用及其特性
@add_start_docstrings(
    "SegGpt model with a decoder on top for one-shot image segmentation.",
    SEGGPT_START_DOCSTRING,
)
# 定义 SegGptForImageSegmentation 类，继承自 SegGptPreTrainedModel
class SegGptForImageSegmentation(SegGptPreTrainedModel):
    
    # 初始化方法，接受一个 SegGptConfig 类型的参数 config
    def __init__(self, config: SegGptConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将参数 config 存储在实例的 config 属性中
        self.config = config

        # 使用给定的 config 创建 SegGptModel 实例，并赋值给 self.model
        self.model = SegGptModel(config)
        # 使用给定的 config 创建 SegGptDecoder 实例，并赋值给 self.decoder
        self.decoder = SegGptDecoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数，执行模型的前向计算
    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SegGptImageSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,  # 图像像素值张量
        prompt_pixel_values: torch.Tensor,  # 提示像素值张量
        prompt_masks: torch.Tensor,  # 提示掩码张量
        bool_masked_pos: Optional[torch.BoolTensor] = None,  # 可选的布尔类型掩码位置张量
        feature_ensemble: Optional[bool] = None,  # 可选的特征合集标志
        embedding_type: Optional[str] = None,  # 可选的嵌入类型
        labels: Optional[torch.FloatTensor] = None,  # 可选的标签张量
        output_attentions: Optional[bool] = None,  # 可选的注意力输出标志
        output_hidden_states: Optional[bool] = None,  # 可选的隐藏状态输出标志
        return_dict: Optional[bool] = None,  # 可选的返回字典标志
```