# `.\models\x_clip\modeling_x_clip.py`

```py
# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Team. All rights reserved.
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
""" PyTorch X-CLIP model."""

# Import necessary modules and functions
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# Importing from within the package
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# Initialize logger for this module
logger = logging.get_logger(__name__)

# Default model checkpoint for documentation purposes
_CHECKPOINT_FOR_DOC = "microsoft/xclip-base-patch32"

# List of pre-trained model archives available for X-CLIP
XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/xclip-base-patch32",
    # See all X-CLIP models at https://huggingface.co/models?filter=x-clip
]

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss given logits.

    Args:
        logits (torch.Tensor): The logits from the model.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->x_clip
def x_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the X-CLIP loss given similarity scores.

    Args:
        similarity (torch.Tensor): The similarity scores between captions and images.

    Returns:
        torch.Tensor: The computed X-CLIP loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())  # Transpose for image loss computation
    return (caption_loss + image_loss) / 2.0


@dataclass
class XCLIPOutput(ModelOutput):
    """
    Placeholder class for model outputs specific to X-CLIP.
    """
    # The class itself is currently empty but serves as a placeholder.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity.
        logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
        video_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`].
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """

    # Optional attributes initialized to None
    loss: Optional[torch.FloatTensor] = None
    logits_per_video: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    video_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    mit_output: BaseModelOutputWithPooling = None

    # Method to convert instance to a tuple, handling special cases for complex types
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # Return self[k] for basic attributes, or convert and return tuple for complex attributes
            self[k]
            if k not in ["text_model_output", "vision_model_output", "mit_output"]
            else getattr(self, k).to_tuple()  # Convert complex attribute to tuple
            for k in self.keys()  # Iterate through all attribute keys
        )
# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->XCLIP
class XCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 定义一个可学习的类别嵌入向量
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 定义用于提取图像补丁特征的二维卷积层
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中补丁的数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # 定义一个位置嵌入层，用于表示每个位置的特征
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # 注册一个位置 id 的缓冲区张量，用于表示序列中每个位置的索引
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype

        # 使用卷积层提取图像补丁的特征表示
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 将类别嵌入向量扩展到每个样本
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        # 将类别嵌入向量和图像补丁特征连接起来作为最终的视觉嵌入表示
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # 添加位置嵌入到最终的视觉嵌入表示中
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->XCLIP
class XCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: XCLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 定义一个用于词嵌入的嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)

        # 定义一个用于位置嵌入的嵌入层
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 注册一个位置 id 的缓冲区张量，用于表示序列中每个位置的索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果没有提供位置 id，则使用预定义的位置 id
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果没有提供嵌入的输入，则使用输入 id 来获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 使用位置嵌入层获取位置嵌入
        position_embeddings = self.position_embedding(position_ids)

        # 将词嵌入和位置嵌入相加作为最终的文本嵌入表示
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->XCLIP
# 定义一个名为 XCLIPAttention 的类，表示从论文 'Attention Is All You Need' 中的多头注意力机制
class XCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 获取隐藏大小作为嵌入维度
        self.num_heads = config.num_attention_heads  # 获取注意力头数
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5  # 缩放因子
        self.dropout = config.attention_dropout  # 注意力机制中的 dropout 概率

        # 线性变换层，用于查询、键、值和输出的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 重新形状张量以便多头注意力计算
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，执行多头注意力计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,



# 定义一个名为 XCLIPMLP 的类，从 CLIP 模型中复制，表示多层感知机（MLP）部分
class XCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 激活函数选择
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个全连接层

    # 前向传播函数，执行全连接层的计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 第一层全连接
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 第二层全连接
        return hidden_states  # 返回计算结果



# 定义一个名为 XCLIPEncoderLayer 的类，从 CLIP 模型中复制，表示编码器层
class XCLIPEncoderLayer(nn.Module):
    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 获取隐藏大小作为嵌入维度
        self.self_attn = XCLIPAttention(config)  # 自注意力层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化层
        self.mlp = XCLIPMLP(config)  # 多层感知机层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化层

    # 前向传播函数，执行编码器层的计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        """
        定义了一个方法，接收以下参数并返回结果的元组：
        - hidden_states (`torch.FloatTensor`): 形状为 `(batch, seq_len, embed_dim)` 的输入层张量
        - attention_mask (`torch.FloatTensor`): 形状为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，
          其中填充元素用非常大的负值表示
        - output_attentions (`bool`, *可选*): 是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 获取更多细节。
        """
        residual = hidden_states  # 保存输入 hidden_states 作为残差连接的基础

        hidden_states = self.layer_norm1(hidden_states)  # 应用第一个层归一化

        # 使用 self_attn 处理 hidden_states，接收 attention_mask 和 output_attentions 作为参数
        # 返回处理后的 hidden_states 和可能的注意力权重 attn_weights
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states  # 将残差连接应用于处理后的 hidden_states

        residual = hidden_states  # 更新残差连接的基础为当前的 hidden_states

        hidden_states = self.layer_norm2(hidden_states)  # 应用第二个层归一化

        hidden_states = self.mlp(hidden_states)  # 应用 MLP 层

        hidden_states = residual + hidden_states  # 再次将残差连接应用于处理后的 hidden_states

        outputs = (hidden_states,)  # 将最终处理后的 hidden_states 存储在 outputs 中

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要返回 attentions，则将 attn_weights 添加到 outputs 中

        return outputs  # 返回包含 hidden_states 和可能的 attn_weights 的元组作为输出
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->XCLIP
class XCLIPDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class XCLIPVisionEncoderLayer(nn.Module):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.hidden_size

        # Message passing components
        self.message_fc = nn.Linear(self.embed_dim, self.embed_dim)  # Linear transformation for message passing
        self.message_ln = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Layer normalization for messages
        self.message_attn = XCLIPAttention(config)  # Attention mechanism for message passing

        # Drop path implementation
        self.drop_path = XCLIPDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        # Self-attention mechanism
        self.self_attn = XCLIPAttention(config)  # Self-attention mechanism
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Layer normalization after self-attention

        # MLP (Feedforward neural network)
        self.mlp = XCLIPMLP(config)  # Multilayer perceptron for feature transformation
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Layer normalization after MLP

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                输入到层的隐藏状态张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
                注意力掩码张量，大小为 `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值表示。
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
                文本模型的因果关注掩码。掩码值选定在 `[0, 1]` 之间：
                - 1 表示 **未屏蔽** 的标记，
                - 0 表示 **屏蔽** 的标记。
                [什么是注意力掩码?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                是否返回所有注意力层的注意力张量。详细信息请参见返回的张量下的 `attentions`。
        """
        # 计算输入张量的维度
        batch_time, seq_length, hidden_size = hidden_states.size()
        # 计算批次大小
        batch_size = batch_time // self.num_frames
        # 提取第一个时间步的隐藏状态，通过全连接层生成消息标记
        msg_token = self.message_fc(hidden_states[:, 0, :])
        # 调整形状以匹配批次和帧数
        msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)

        # 对消息标记应用注意力操作，并通过随机丢弃路径进行正则化
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token))[0])
        # 添加虚拟序列维度
        msg_token = msg_token.view(-1, 1, hidden_size)

        # 将消息标记连接到原始输入张量中
        hidden_states = torch.cat([hidden_states, msg_token], dim=1)

        # 保存残差连接
        residual = hidden_states

        # 对连接后的张量进行层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 应用自注意力机制，并返回注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 裁剪处理后的张量，恢复原始的序列长度
        hidden_states = hidden_states[:, :seq_length, :]

        # 保存残差连接
        residual = hidden_states
        # 对连接后的张量再次进行层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 应用多层感知机层
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 输出结果为包含处理后的隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将它们添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义一个名为 XCLIPPreTrainedModel 的类，继承自 PreTrainedModel，用于处理权重初始化和预训练模型的下载与加载的简单接口。
class XCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定该类的配置类为 XCLIPConfig，用于处理模型配置信息
    config_class = XCLIPConfig
    # 基础模型的前缀名称为 "x_clip"，用于标识模型的基础结构
    base_model_prefix = "x_clip"
    # 支持梯度检查点技术，允许在模型训练时进行梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 从配置中获取初始化因子
        factor = self.config.initializer_factor
        
        # 如果 module 是 XCLIPTextEmbeddings 类型
        if isinstance(module, XCLIPTextEmbeddings):
            # 初始化 token_embedding 的权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 初始化 position_embedding 的权重
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        
        # 如果 module 是 XCLIPVisionEmbeddings 类型
        elif isinstance(module, XCLIPVisionEmbeddings):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            # 初始化 class_embedding 的权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 初始化 patch_embedding 的权重
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 初始化 position_embedding 的权重
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        
        # 如果 module 是 XCLIPAttention 类型
        elif isinstance(module, XCLIPAttention):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算输出投影的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化 q_proj 的权重
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            # 初始化 k_proj 的权重
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            # 初始化 v_proj 的权重
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            # 初始化 out_proj 的权重
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        
        # 如果 module 是 XCLIPMLP 类型
        elif isinstance(module, XCLIPMLP):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化 fc1 的权重
            nn.init.normal_(module.fc1.weight, std=fc_std)
            # 初始化 fc2 的权重
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        
        # 如果 module 是 XCLIPModel 类型
        elif isinstance(module, XCLIPModel):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            # 初始化 text_projection 的权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            # 初始化 visual_projection 的权重
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )
            # 初始化 prompts_visual_projection 的权重
            nn.init.normal_(module.prompts_visual_projection, mean=0.0, std=module.vision_embed_dim**-0.5 * factor)
        
        # 如果 module 是 XCLIPMultiframeIntegrationTransformer 类型
        elif isinstance(module, XCLIPMultiframeIntegrationTransformer):
            # 初始化 position_embedding 的权重
            nn.init.normal_(module.position_embedding, std=self.config.initializer_factor)
        
        # 如果 module 是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)
        
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            # 如果有偏置项，将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
# X_CLIP_START_DOCSTRING 是一个包含模型描述信息的原始字符串文档
X_CLIP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`XCLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# X_CLIP_TEXT_INPUTS_DOCSTRING 是一个关于文本输入参数的文档字符串
X_CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# X_CLIP_VISION_INPUTS_DOCSTRING 是一个空字符串，暂时未包含内容
X_CLIP_VISION_INPUTS_DOCSTRING = r"""
"""
    Args:
        # `pixel_values` 是一个 torch.FloatTensor，表示图像的像素值，形状为 `(batch_size, num_channels, height, width)`
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        
        # 是否输出所有注意力层的注意力张量，默认为 False
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        # 是否输出所有隐藏层的隐藏状态，默认为 False
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        # 是否返回一个 `utils.ModelOutput` 对象而不是普通的元组，默认为 False
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
X_CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->XCLIP
class XCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.config = config
        # 创建一个由多个 XCLIPEncoderLayer 组成的列表，数量为 config.num_hidden_layers
        self.layers = nn.ModuleList([XCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 XCLIPTextTransformer 类，继承自 nn.Module，用于文本转换任务的模型
class XCLIPTextTransformer(nn.Module):
    # 初始化函数，接受一个 XCLIPTextConfig 类型的配置对象作为参数
    def __init__(self, config: XCLIPTextConfig):
        super().__init__()
        # 将配置对象保存在模型中
        self.config = config
        # 根据配置对象中的 hidden_size 参数设置嵌入维度
        embed_dim = config.hidden_size
        # 创建 XCLIPTextEmbeddings 对象，用于文本的嵌入表示
        self.embeddings = XCLIPTextEmbeddings(config)
        # 创建 XCLIPEncoder 对象，用于编码文本信息
        self.encoder = XCLIPEncoder(config)
        # 创建最终的 LayerNorm 层，用于规范化最终输出的特征
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数，处理输入数据并返回模型的输出
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 如果没有显式提供output_attentions，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式提供output_hidden_states，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式提供return_dict，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有提供input_ids，则抛出数值错误异常
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取input_ids的形状
        input_shape = input_ids.size()
        # 将input_ids视图调整为二维张量
        input_ids = input_ids.view(-1, input_shape[-1])

        # 使用self.embeddings嵌入输入的input_ids和可选的position_ids
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # X_CLIP的文本模型使用因果掩码，在这里准备它
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        # 创建四维因果注意力掩码
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # 如果提供了attention_mask，则将其扩展为四维张量
        if attention_mask is not None:
            # 将二维的attention_mask扩展为四维的
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 使用encoder对输入进行编码，得到encoder_outputs
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后隐藏状态进行最终的层归一化
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 从最后隐藏状态中提取汇总输出（pooled_output）
        # pooled_output是从每个序列中eot embedding（end-of-token）中获取特征
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        # 如果return_dict为False，则返回一个元组，包括最后隐藏状态、汇总输出和其他输出状态
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果return_dict为True，则返回一个BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class XCLIPTextModel(XCLIPPreTrainedModel):
    config_class = XCLIPTextConfig

    def __init__(self, config: XCLIPTextConfig):
        super().__init__(config)
        # 使用传入的配置初始化文本模型
        self.text_model = XCLIPTextTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的输入嵌入层（token embedding）
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        # 设置文本模型的输入嵌入层（token embedding）
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```
        >>> from transformers import AutoTokenizer, XCLIPTextModel

        >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
        使用文本模型处理传入的参数，并返回处理结果
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class XCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPVisionEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.config = config
        # 创建多层视觉编码器，每层是一个XCLIPVisionEncoderLayer实例
        self.layers = nn.ModuleList([XCLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 视觉编码器的前向传播方法，处理输入嵌入，注意力掩码等，返回处理结果
        pass  # pass语句表示这里暂时没有额外的操作，仅作为占位符使用


class XCLIPVisionTransformer(nn.Module):
    """
    This corresponds to the `CrossFrameCommunicationTransformer` class in the original implementation.
    """
    # 初始化方法，接受一个配置对象 config: XCLIPVisionConfig，并调用父类的初始化方法
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        # 将传入的配置对象保存到实例属性中
        self.config = config
        # 从配置对象中获取隐藏层的维度，并保存为 embed_dim
        embed_dim = config.hidden_size

        # 创建 XCLIPVisionEmbeddings 对象，并保存到实例属性中
        self.embeddings = XCLIPVisionEmbeddings(config)
        # 创建 LayerNorm 层，用于前处理，对输入进行归一化，eps 参数来自配置对象
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建 XCLIPVisionEncoder 对象，并保存到实例属性中
        self.encoder = XCLIPVisionEncoder(config)
        # 创建 LayerNorm 层，用于后处理，对输出进行归一化，eps 参数来自配置对象
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 前向传播方法，接受像素值 pixel_values 和一些可选参数，返回模型输出
    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 如果 output_attentions 参数不为 None，则使用其值；否则使用配置对象中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数不为 None，则使用其值；否则使用配置对象中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数不为 None，则使用其值；否则使用配置对象中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将像素值传入嵌入层，并获取隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行前处理，应用 LayerNorm
        hidden_states = self.pre_layernorm(hidden_states)

        # 将前处理后的隐藏状态传入编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器输出中获取最后一层隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 从最后一层隐藏状态中获取池化输出，通常是第一个位置的输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后处理，应用 LayerNorm
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不要求返回字典形式的结果，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典形式的结果，则创建 BaseModelOutputWithPooling 对象，并返回
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class XCLIPVisionModel(XCLIPPreTrainedModel):
    # 指定配置类为XCLIPVisionConfig
    config_class = XCLIPVisionConfig
    # 主要输入名称为"pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: XCLIPVisionConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 初始化视觉模型为XCLIPVisionTransformer的实例
        self.vision_model = XCLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型中的补丁嵌入层作为输入嵌入层
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 正向传播函数定义
        # pixel_values: 像素值作为输入
        # output_attentions: 是否输出注意力
        # output_hidden_states: 是否输出隐藏状态
        # return_dict: 是否返回字典形式结果
        ...



class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    这对应于原始实现中的`MultiframeIntegrationTransformer`类。
    """

    def __init__(self, config: XCLIPVisionConfig):
        # 初始化函数
        super().__init__()
        # 定义位置嵌入为可学习参数
        self.position_embedding = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        # 使用XCLIPEncoder对数据进行编码
        self.encoder = XCLIPEncoder(config)

    def forward(
        self,
        hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 正向传播函数定义
        residual = hidden_states

        # 添加位置嵌入到隐藏状态中
        hidden_states = hidden_states + self.position_embedding

        # 调用编码器对输入进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 将最后隐藏状态转换为与输入相同的数据类型，并加上残差连接
        last_hidden_state = last_hidden_state.type(hidden_states.dtype) + residual

        # 计算池化输出，取平均值
        pooled_output = last_hidden_state.mean(dim=1, keepdim=False)

        # 如果不返回字典形式结果，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带池化的基础模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class XCLIPCrossAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""



# XCLIPCrossAttention类定义了一个多头注意力机制，来源于'Attention Is All You Need'论文
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.prompt_num_attention_heads  # 从配置中获取注意力头的数量

        dim = config.projection_dim  # 从配置中获取投影维度
        head_dim = dim // self.num_heads  # 计算每个注意力头的维度
        self.scale = head_dim**-0.5  # 缩放因子，用于缩放注意力权重

        # 初始化查询、键、值的线性投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(config.prompt_attention_dropout)  # 注意力分数的dropout层
        self.proj = nn.Linear(dim, dim)  # 最终输出的线性投影层
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)  # 最终输出的dropout层

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        """调整张量形状以便注意力计算"""
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, queries, keys, values):
        """模型的前向传播方法"""
        batch_size, query_seq_len, hidden_size = queries.shape
        batch_size, key_seq_len, hidden_size = keys.shape

        # 对查询、键、值进行线性投影，并调整形状以适应注意力计算的需求
        queries = (
            self.q_proj(queries)
            .reshape(batch_size, query_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(keys)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        values = (
            self.v_proj(values)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # 计算注意力权重并进行缩放
        attn = (queries @ keys.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 对注意力权重进行softmax归一化
        attn = self.attn_drop(attn)  # 对注意力分数进行dropout

        # 通过注意力权重加权值向量，然后将结果重新整形为最终输出形状
        x = (attn @ values).transpose(1, 2).reshape(batch_size, query_seq_len, hidden_size)
        x = self.proj(x)  # 最终输出的线性投影
        x = self.proj_drop(x)  # 对最终输出进行dropout
        return x
class PromptGeneratorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.projection_dim
        # 初始化跨媒体注意力层
        self.cross_attn = XCLIPCrossAttention(config)
        # 第一层归一化
        self.norm1 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        # 第三层归一化
        self.norm3 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        # 多层感知机模型
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 线性层，扩展维度
            ACT2FN[config.prompt_hidden_act],  # 激活函数
            nn.Dropout(config.prompt_attention_dropout),  # 随机丢弃以减少过拟合
            nn.Linear(embed_dim * 4, embed_dim),  # 线性层，降低维度
        )

    def forward(self, x, visual):
        # 使用跨媒体注意力层处理文本和视觉输入
        x = x + self.cross_attn(self.norm1(x), visual, visual)
        # 使用多层感知机处理更新后的文本表示
        x = x + self.mlp(self.norm3(x))
        return x


class XCLIPPromptGenerator(nn.Module):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.projection_dim
        # 规范化层，用于视觉输入
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.vision_config.layer_norm_eps)
        # 多个生成层组成的解码器
        self.decoder = nn.ModuleList([PromptGeneratorLayer(config) for _ in range(config.prompt_layers)])
        # 系数 alpha，用于加权输出文本表示
        self.alpha = nn.Parameter(torch.ones(embed_dim) * config.prompt_alpha)

    def forward(self, text, visual):
        # 规范化视觉输入
        visual = self.layernorm(visual)
        # 逐层使用生成层处理文本和视觉输入
        for layer in self.decoder:
            text = layer(text, visual)

        # 使用 alpha 系数加权输出文本表示
        return self.alpha * text


@add_start_docstrings(X_CLIP_START_DOCSTRING)
class XCLIPModel(XCLIPPreTrainedModel):
    config_class = XCLIPConfig
    # 初始化方法，接受一个XCLIPConfig类型的参数config
    def __init__(self, config: XCLIPConfig):
        # 调用父类的初始化方法，传入config参数
        super().__init__(config)

        # 检查config.text_config是否为XCLIPTextConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.text_config, XCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type XCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查config.vision_config是否为XCLIPVisionConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.vision_config, XCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type XCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从config中获取text_config和vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置对象的投影维度为config.projection_dim
        self.projection_dim = config.projection_dim
        # 设置对象的文本嵌入维度为text_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        # 设置对象的视觉嵌入维度为vision_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建文本模型，使用XCLIPTextTransformer类，并传入text_config作为参数
        self.text_model = XCLIPTextTransformer(text_config)
        # 创建视觉模型，使用XCLIPVisionTransformer类，并传入vision_config作为参数
        self.vision_model = XCLIPVisionTransformer(vision_config)

        # 创建视觉投影层，将视觉嵌入维度映射到投影维度，不使用偏置
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # 创建文本投影层，将文本嵌入维度映射到投影维度，不使用偏置
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # 创建一个可学习参数，作为logit的尺度初始化值，使用config中的logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 创建视觉提示的LayerNorm层，输入维度为视觉嵌入维度，使用config中的layer_norm_eps作为epsilon值
        self.prompts_visual_layernorm = nn.LayerNorm(self.vision_embed_dim, eps=config.vision_config.layer_norm_eps)
        # 创建一个可学习参数，形状为[视觉嵌入维度, 投影维度]，用于视觉提示的投影
        self.prompts_visual_projection = nn.Parameter(torch.randn(self.vision_embed_dim, self.projection_dim))

        # 复制vision_config创建一个新的配置mit_config，并修改其中的部分属性
        mit_config = copy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        # 创建XCLIPMultiframeIntegrationTransformer对象，使用修改后的mit_config作为参数
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)

        # 创建XCLIPPromptGenerator对象，使用config作为参数
        self.prompts_generator = XCLIPPromptGenerator(config)

        # 调用post_init方法，完成权重初始化和最终处理
        self.post_init()

    # 为模型的前向传播函数get_text_features添加文档字符串，使用X_CLIP_TEXT_INPUTS_DOCSTRING作为注释模板
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPTextModel`].

        Examples:

        ```
        >>> from transformers import AutoTokenizer, AutoModel

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        return text_embeds

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    def get_video_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            video_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The video embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPVideoModel`].

        Examples:

        ```
        >>> from transformers import AutoTokenizer, AutoModel

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> video_inputs = torch.randn(2, 3, 224, 224)  # Example pixel values tensor
        >>> video_features = model.get_video_features(pixel_values=video_inputs)
        ```"""
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        video_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        video_embeds = video_outputs[1]
        video_embeds = self.video_projection(video_embeds)

        return video_embeds

    @add_start_docstrings_to_model_forward(X_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XCLIPOutput, config_class=XCLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[XCLIPOutput, Tuple[torch.FloatTensor]]:
        r"""
        Returns:
            Union[XCLIPOutput, Tuple[torch.FloatTensor]]: Depending on the configuration, this method can return either
            a single XCLIPOutput object or a tuple containing a torch.FloatTensor.
        """
```