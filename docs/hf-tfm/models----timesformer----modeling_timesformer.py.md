# `.\models\timesformer\modeling_timesformer.py`

```
# coding=utf-8
# 文件编码声明，确保支持 UTF-8 编码格式
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
# 版权声明及保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 版本授权许可
# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据"原样"分发本软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示保证或条件
# See the License for the specific language governing permissions and
# 请参阅许可证了解特定的语言规定和权限
# limitations under the License.
# 许可证下的限制

""" PyTorch TimeSformer model."""
# PyTorch TimeSformer 模型声明

import collections
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig

# 导入必要的库和模块

logger = logging.get_logger(__name__)
# 获取模块的日志记录器对象

_CONFIG_FOR_DOC = "TimesformerConfig"
_CHECKPOINT_FOR_DOC = "facebook/timesformer"

TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/timesformer-base-finetuned-k400",
    # See all TimeSformer models at https://huggingface.co/models?filter=timesformer
]
# Timesformer 预训练模型的存档列表

# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L155
class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""
    # 图像转换为补丁嵌入的模块声明

    def __init__(self, config):
        super().__init__()
        # 调用父类构造函数初始化模块

        image_size = config.image_size
        patch_size = config.patch_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 计算图像中的补丁数目

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 设置模块的图像尺寸、补丁尺寸和补丁数目属性

        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        # 使用二维卷积将图像像素映射到补丁嵌入空间

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        # 重塑输入张量以适应卷积层的输入要求

        embeddings = self.projection(pixel_values)
        # 将像素值投影到嵌入空间
        patch_width = embeddings.size(-1)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 对嵌入进行扁平化并转置以便进一步处理
        return embeddings, num_frames, patch_width
        # 返回嵌入向量、帧数和补丁宽度信息


class TimesformerEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """
    # 构建补丁和位置嵌入的模块声明
    # 初始化函数，用于初始化一个新的实例对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 从配置中获取帧数作为时间维度
        num_frames = config.num_frames
        # 从配置中获取隐藏层的dropout率
        drop_rate = config.hidden_dropout_prob
        # 从配置中获取注意力机制的类型
        attention_type = config.attention_type

        # 将注意力机制的类型保存到实例对象中
        self.attention_type = attention_type
        # 使用TimesformerPatchEmbeddings类初始化补丁嵌入层
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        # 计算补丁的数量并保存到实例对象中
        self.num_patches = self.patch_embeddings.num_patches

        # 位置嵌入部分
        # 创建一个可学习的用于分类的令牌（CLS token）参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 创建一个可学习的位置嵌入参数，考虑到补丁数和一个额外的CLS token
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        # 应用dropout到位置嵌入参数
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 如果注意力机制不是"space_only"，则初始化时间嵌入部分
        if attention_type != "space_only":
            # 创建一个可学习的时间嵌入参数，考虑到帧数
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            # 应用dropout到时间嵌入参数
            self.time_drop = nn.Dropout(p=drop_rate)
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]  # 获取输入张量的批量大小

        # create patch embeddings
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)
        # 生成图像的补丁嵌入表示，并获取补丁的帧数和宽度信息

        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        # 扩展类标记以匹配嵌入张量的批量大小
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # 将类标记与补丁嵌入张量连接起来

        # resizing the positional embeddings in case they don't match the input at inference
        if embeddings.size(1) != self.position_embeddings.size(1):
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            # 提取类标记的位置嵌入，并调整维度以匹配嵌入张量的格式
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            # 提取其它位置嵌入并转置维度以匹配嵌入张量的格式
            patch_num = int(other_pos_embed.size(2) ** 0.5)
            patch_height = embeddings.size(1) // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.size(2), patch_num, patch_num)
            # 重塑其它位置嵌入以匹配嵌入张量的形状
            new_pos_embed = nn.functional.interpolate(
                other_pos_embed, size=(patch_height, patch_width), mode="nearest"
            )
            # 使用最近邻插值调整其它位置嵌入的尺寸以匹配补丁的形状
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            # 将调整后的位置嵌入与类标记位置嵌入连接
            embeddings = embeddings + new_pos_embed
            # 将调整后的位置嵌入添加到补丁嵌入张量中
        else:
            embeddings = embeddings + self.position_embeddings
            # 否则，直接将位置嵌入添加到补丁嵌入张量中

        embeddings = self.pos_drop(embeddings)
        # 对位置嵌入后的张量进行dropout操作

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            # 提取类标记以处理时间嵌入
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = (
                embeddings.reshape(batch_size, num_frames, patch_height, patch_width)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * patch_height, num_frames, patch_width)
            )
            # 重新排列张量以适应时间嵌入的处理
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)
                new_time_embeddings = nn.functional.interpolate(time_embeddings, size=(num_frames), mode="nearest")
                new_time_embeddings = new_time_embeddings.transpose(1, 2)
                # 调整时间嵌入的尺寸以匹配帧数
                embeddings = embeddings + new_time_embeddings
                # 将调整后的时间嵌入添加到嵌入张量中
            else:
                embeddings = embeddings + self.time_embeddings
                # 否则，直接将时间嵌入添加到嵌入张量中

            embeddings = self.time_drop(embeddings)
            # 对时间嵌入后的张量进行dropout操作
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(
                batch_size, patch_height * num_frames, patch_width
            )
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            # 将类标记与处理后的张量连接

        return embeddings
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
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TimeSformer
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L57
class TimesformerSelfAttention(nn.Module):
    def __init__(self, config: TimesformerConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        # Linear transformation for Query, Key, Value (QKV) using fully connected layer
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        # Dropout layer for attention scores
        self.attn_drop = nn.Dropout(attention_dropout_prob)
    # 定义一个前向传播函数，用于处理输入的隐藏状态和可能输出注意力分布
    def forward(self, hidden_states, output_attentions: bool = False):
        # 获取输入隐藏状态的批量大小、隐藏大小和通道数量
        batch_size, hidden_size, num_channels = hidden_states.shape
        
        # 将隐藏状态传入QKV层，将输出reshape成(batch_size, hidden_size, 3, num_heads, num_channels // num_heads)，并进行维度转置
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数，通过query和key的乘积后乘以缩放比例self.scale
        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        # 对注意力分数进行softmax操作，使得其在最后一个维度上的和为1
        attention_probs = attention_probs.softmax(dim=-1)
        # 对注意力分数应用dropout操作，以防止过拟合
        attention_probs = self.attn_drop(attention_probs)

        # 计算上下文向量，通过attention_probs和value的乘积，然后进行维度转置和reshape操作
        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, hidden_size, num_channels)

        # 如果指定了output_attentions为True，则同时返回上下文层和注意力分布；否则，只返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回最终的输出结果
        return outputs
# TimesformerSelfOutput 类定义，继承自 nn.Module
class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入和输出大小为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 dropout 层，以 config.hidden_dropout_prob 为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收一个名为 hidden_states 的张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# TimesformerAttention 类定义，继承自 nn.Module
class TimeSformerAttention(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 创建 TimesformerSelfAttention 对象，使用 config 参数
        self.attention = TimesformerSelfAttention(config)
        # 创建 TimesformerSelfOutput 对象，使用 config 参数
        self.output = TimesformerSelfOutput(config)

    # 前向传播函数，接收 hidden_states 张量和 output_attentions 布尔值参数，返回处理后的张量或元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 self.attention 的前向传播方法，处理 hidden_states 和 output_attentions
        self_outputs = self.attention(hidden_states, output_attentions)

        # 调用 self.output 的前向传播方法，处理 self_outputs 的第一个元素
        attention_output = self.output(self_outputs[0])

        # 如果输出注意力信息，则在输出元组中添加注意力张量
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# TimesformerIntermediate 类定义，继承自 nn.Module
class TimesformerIntermediate(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入为 config.hidden_size，输出为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建一个 dropout 层，以 config.hidden_dropout_prob 为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 根据 config.hidden_act 的类型选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接收 hidden_states 张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数处理处理后的张量
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对处理后的张量应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# TimesformerOutput 类定义，继承自 nn.Module
class TimesformerOutput(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入为 config.intermediate_size，输出为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 dropout 层，以 config.hidden_dropout_prob 为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收 hidden_states 张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# TimesformerLayer 类定义，继承自 nn.Module
# 代码未完整提供，注释省略
class TimesformerLayer(nn.Module):
    def __init__(self, config: TimesformerConfig, layer_index: int) -> None:
        super().__init__()

        # 从配置中获取注意力类型
        attention_type = config.attention_type

        # 根据规则生成随机深度路径的下降率列表
        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # 随机深度路径的下降率衰减规则
        drop_path_rate = drop_path_rates[layer_index]

        # 如果下降率大于0，则使用自定义的时间变换下降路径；否则使用恒等映射
        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        # 初始化注意力机制、中间层和输出层
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        
        # 在层归一化之前和之后使用配置中定义的层归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 存储配置信息和注意力类型，如果类型不在预定义的范围内，则引发值错误
        self.config = config
        self.attention_type = attention_type
        if attention_type not in ["divided_space_time", "space_only", "joint_space_time"]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # 如果注意力类型为"divided_space_time"，则初始化时间注意力相关参数
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)
class TimesformerEncoder(nn.Module):
    # TimesformerEncoder 类，用于实现 Timesformer 模型的编码器部分
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.config = config
        # 初始化多层 TimesformerLayer 模块组成的列表
        self.layer = nn.ModuleList([TimesformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 是否开启梯度检查点

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空的元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个 TimesformerLayer 层进行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 如果开启了梯度检查点且处于训练模式，则使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不需要返回字典形式的输出，则返回所有非空的结果元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 BaseModelOutput 对象，包含最终的输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # Timesformer 预训练模型的抽象类，处理权重初始化以及预训练模型下载和加载的简单接口

    config_class = TimesformerConfig  # Timesformer 模型配置类
    base_model_prefix = "timesformer"  # Timesformer 模型的基础名称前缀
    main_input_name = "pixel_values"  # Timesformer 模型的主要输入名称
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def _init_weights(self, module):
        # 初始化模型权重的函数
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或卷积层，则使用截断正态分布初始化权重
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # 如果有偏置，初始化为常数 0
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，则初始化偏置为常数 0，权重为常数 1
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddings):
            # 如果是 TimesformerEmbeddings 类型，则分别初始化 cls_token 和 position_embeddings
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)


TIMESFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    Parameters:
        config ([`TimesformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
TIMESFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.preprocess`] for details.

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
    "The bare TimeSformer Model transformer outputting raw hidden-states without any specific head on top.",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerModel(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize embeddings and encoder based on provided configuration
        self.embeddings = TimesformerEmbeddings(config)
        self.encoder = TimesformerEncoder(config)

        # Layer normalization for post-encoder processing
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the patch embeddings used in the model's input layer.
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (dict): dict of {layer_num: list of heads to prune in this layer}
                                  See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the Timesformer Model.

        Args:
            pixel_values (torch.FloatTensor): Pixel values of shape `(batch_size, num_frames, num_channels, height, width)`.
            output_attentions (bool, optional): Whether to return attentions tensors of all attention layers.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.
            return_dict (bool, optional): Whether to return a ModelOutput instead of a plain tuple.

        Returns:
            BaseModelOutput or tuple:
                A BaseModelOutput (if return_dict=True) or a tuple of torch.FloatTensor containing various model outputs.
        """
        # Implementation of forward pass goes here
        pass


@add_start_docstrings(
    """TimeSformer Model transformer with a video classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.""",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerForVideoClassification(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize number of labels and base Timesformer model
        self.num_labels = config.num_labels
        self.timesformer = TimesformerModel(config)

        # Classifier head for video classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the TimesformerForVideoClassification model.

        Args:
            pixel_values (torch.FloatTensor): Pixel values of shape `(batch_size, num_frames, num_channels, height, width)`.
            output_attentions (bool, optional): Whether to return attentions tensors of all attention layers.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.
            return_dict (bool, optional): Whether to return a ModelOutput instead of a plain tuple.

        Returns:
            BaseModelOutput or tuple:
                A BaseModelOutput (if return_dict=True) or a tuple of torch.FloatTensor containing various model outputs.
        """
        # Implementation of forward pass goes here
        pass
    # 使用装饰器替换返回文档字符串，设置输出类型为ImageClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数，接受多个参数：
    # pixel_values: 可选的torch.Tensor，表示输入像素值
    # labels: 可选的torch.Tensor，表示标签数据
    # output_attentions: 可选的bool值，控制是否输出注意力权重
    # output_hidden_states: 可选的bool值，控制是否输出隐藏状态
    # return_dict: 可选的bool值，控制是否返回字典形式的结果
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```