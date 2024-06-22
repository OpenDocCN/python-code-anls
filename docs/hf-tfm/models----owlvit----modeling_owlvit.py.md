# `.\transformers\models\owlvit\modeling_owlvit.py`

```py
# coding=utf-8
# 此代码文件使用 utf-8 编码

# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
# 该代码版权归 Google AI 和 HuggingFace 团队所有

#
# Licensed under the Apache License, Version 2.0 (the "License");
# # 根据 Apache 许可证 2.0 版本获得授权，只能在遵守许可证的条件下使用

# you may not use this file except in compliance with the License.
# 你只能在遵守许可证的前提下使用该文件

# You may obtain a copy of the License at
# 你可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，根据许可证分发的软件

# distributed under the License is distributed on an "AS IS" BASIS,
# 是在"原样"的基础上分发的，

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论是明示还是暗示的担保或条件

# See the License for the specific language governing permissions and
# 请参阅许可证中规定的特定于语言的权限和
# limitations under the License.
# 限制。

""" PyTorch OWL-ViT model."""
# 此文件定义了 PyTorch 版本的 OWL-ViT 模型

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/owlvit-base-patch32"

# See all OwlViT models at https://huggingface.co/models?filter=owlvit
OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlvit-base-patch32",
    "google/owlvit-base-patch16",
    "google/owlvit-large-patch14",
]


# Copied from transformers.models.clip.modeling_clip.contrastive_loss with clip->owlvit
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 计算对比损失，即使用交叉熵损失函数
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->owlvit
def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算 OWL-ViT 损失函数
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class OwlViTOutput(ModelOutput):
    """
    # 定义 OwlViTOutput 数据类，包含模型的输出
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`OwlViTVisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """

    # 定义可能的输入参数
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 转换对象为元组形式
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果键不是 "text_model_output" 或 "vision_model_output"，则直接获取值；否则，获取相应属性的元组形式
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 将输入张量向更高的等价类型类型转换，以防止乘法导致的数值溢出
def _upcast(t: Tensor) -> Tensor:
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 计算一组边界框的面积，这些边界框由它们的（x1，y1，x2，y2）坐标指定
def box_area(boxes: Tensor) -> Tensor:
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 计算两组边界框的交并比
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# 计算两组边界框的广义交并比
def generalized_box_iou(boxes1, boxes2):
    # 如果边界框是退化的，则提前进行检查，避免出现无限值或非数值结果
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


@dataclass
class OwlViTObjectDetectionOutput(ModelOutput):
    """
    [`OwlViTForObjectDetection`]的输出类型。
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    # 初始化变量，用于存储各种模型输出及损失
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 将各种模型输出及损失转换为元组形式
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果键不是"text_model_output"或"vision_model_output"，则直接取值，否则调用对应对象的to_tuple()方法
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 创建一个名为 OwlViTImageGuidedObjectDetectionOutput 的数据类，继承自 ModelOutput
@dataclass
class OwlViTImageGuidedObjectDetectionOutput(ModelOutput):
    """
   [`OwlViTForObjectDetection.image_guided_detection`] 的输出类型。

    参数:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            所有查询的分类 logits（包括无对象）。
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            所有查询的标准化框坐标，表示为（中心_x，中心_y，宽度，高度）。这些值在 [0, 1] 范围内标准化，相对于批处理中每个单独目标图像的大小（忽略可能的填充）。您可以使用[`~OwlViTImageProcessor.post_process_object_detection`]来获取非标准化的边界框。
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            所有查询的标准化框坐标，表示为（中心_x，中心_y，宽度，高度）。这些值在 [0, 1] 范围内标准化，相对于批处理中每个单独查询图像的大小（忽略可能的填充）。您可以使用[`~OwlViTImageProcessor.post_process_object_detection`]来获取非标准化的边界框。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            [`OwlViTVisionModel`]的池化输出。OWL-ViT将图像表示为一组图像补丁，并为每个补丁计算图像嵌入。
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            [`OwlViTVisionModel`]的池化输出。OWL-ViT将图像表示为一组图像补丁，并为每个补丁计算图像嵌入。
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            所有图像补丁的类嵌入。OWL-ViT将图像表示为一组图像补丁，其中补丁的总数是（image_size / patch_size）**2。
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            [`OwlViTTextModel`]的输出。
        vision_model_output (`BaseModelOutputWithPooling`):
            [`OwlViTVisionModel`]的输出。
    """

    # 初始化默认值为 None 的各个属性
    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 将对象转换为元组的方法
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class OwlViTVisionEmbeddings(nn.Module):
    # 初始化函数，参数为配置对象 OwlViTVisionConfig
    def __init__(self, config: OwlViTVisionConfig):
        # 调用父类初始化函数
        super().__init__()
        # 保存配置对象
        self.config = config
        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 初始化类别嵌入，作为可学习的参数
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # 设置补丁嵌入层，使用卷积操作
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # 计算补丁数量
        self.num_patches = (config.image_size // config.patch_size) ** 2
        # 计算位置编码数量
        self.num_positions = self.num_patches + 1
        # 初始化位置编码层，作为可学习的参数
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册位置编码索引为缓冲区，不参与梯度更新
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播函数，输入为像素数值张量，输出为特征嵌入张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取批量大小
        batch_size = pixel_values.shape[0]
        # 对输入像素值进行补丁嵌入操作
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        # 将补丁嵌入展开并进行维度交换，以便与位置编码进行加和
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类别嵌入至与补丁嵌入相同的维度
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 拼接类别嵌入和补丁嵌入，得到最终特征嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上位置编码得到最终嵌入结果
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回特征嵌入结果
        return embeddings
class OwlViTTextEmbeddings(nn.Module):
    # OwlViTTextEmbeddings 类的构造函数
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()
        # 创建 token embedding 层，用于将输入 token 映射为隐藏表示
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # 创建 position embedding 层，用于编码位置信息
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 创建并注册 position_ids 缓冲区，存储位置索引信息，不会被训练
        # position_ids (1, len position emb) 在序列化时被连续地导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # OwlViTTextEmbeddings 类的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果输入的是 token_ids，获取序列长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果未提供 position_ids，则使用预先注册的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 inputs_embeds，则使用 token_embedding 层将 token_ids 转换为嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入表示
        position_embeddings = self.position_embedding(position_ids)
        # 将 token 嵌入表示和位置嵌入表示相加
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class OwlViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # OwlViTAttention 类的构造函数
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 必须能被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子，用于调节注意力计算中的数值范围
        self.scale = self.head_dim**-0.5
        # dropout 概率，用于注意力计算中的 dropout 操作
        self.dropout = config.attention_dropout

        # 线性变换，将隐藏表示投影到 Q、K、V 空间
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 最终的输出投影
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 将张量形状调整为适用于多头注意力计算的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # OwlViTAttention 类的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 省略了函数体，根据多头注意力机制实现前向传播


# 从 'Attention Is All You Need' 论文中的多头注意力机制
class OwlViTMLP(nn.Module):
    # OwlViTMLP 类的构造函数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个线性变换层
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个线性变换层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    # 定义一个前向传播函数，接受隐藏状态作为输入，返回转换后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层 fc1 对隐藏状态进行线性变换
        hidden_states = self.fc1(hidden_states)
        # 对线性变换后的结果应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 使用全连接层 fc2 对激活后的隐藏状态再进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 返回最终处理过的隐藏状态
        return hidden_states
# OwlViTEncoderLayer 类是从 transformers.models.clip.modeling_clip.CLIPEncoderLayer 复制来的，用于 OwlViT 模型
class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config: OwlViTConfig):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 获取隐藏层大小
        self.embed_dim = config.hidden_size
        # 创建自注意力层
        self.self_attn = OwlViTAttention(config)
        # 创建第一个层归一化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建 MLP 层
        self.mlp = OwlViTMLP(config)
        # 创建第二个层归一化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入作为残差
        residual = hidden_states

        # 通过第一个层归一化层
        hidden_states = self.layer_norm1(hidden_states)
        # 通过自注意力层
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 将自注意力层的输出与残差相加
        hidden_states = residual + hidden_states

        # 保存输入作为残差
        residual = hidden_states
        # 通过第二个层归一化层
        hidden_states = self.layer_norm2(hidden_states)
        # 通过 MLP 层
        hidden_states = self.mlp(hidden_states)
        # 将 MLP 层的输出与残差相加
        hidden_states = residual + hidden_states

        # 输出包括隐藏状态，可选注意力权重
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# OwlViTPreTrainedModel 是一个抽象基类，用于处理权重初始化和加载预训练模型的简单接口
class OwlViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 OwlViTConfig
    config_class = OwlViTConfig
    # 设置基本模型前缀为 "owlvit"
    base_model_prefix = "owlvit"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块包括 OwlViTEncoderLayer
    _no_split_modules = ["OwlViTEncoderLayer"]
    # 定义一个方法，用于初始化指定模块的权重
    def _init_weights(self, module):
        """初始化权重"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果模块是 OwlViTTextEmbeddings，初始化嵌入层的权重
        if isinstance(module, OwlViTTextEmbeddings):
            # 使用正态分布初始化 token 嵌入权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 使用正态分布初始化位置嵌入权重
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果模块是 OwlViTVisionEmbeddings，初始化视觉嵌入的权重
        elif isinstance(module, OwlViTVisionEmbeddings):
            # 使用正态分布初始化 class 嵌入
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 使用正态分布初始化 patch 嵌入权重
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 使用正态分布初始化位置嵌入权重
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果模块是 OwlViTAttention，初始化注意力机制的权重
        elif isinstance(module, OwlViTAttention):
            # 计算注意力输入权重的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算注意力输出权重的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化注意力机制的投影权重
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果模块是 OwlViTMLP，初始化多层感知器的权重
        elif isinstance(module, OwlViTMLP):
            # 计算 MLP 输入权重的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化 MLP 的全连接层权重
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果模块是 OwlViTModel，初始化模型投影的权重
        elif isinstance(module, OwlViTModel):
            # 初始化文本投影权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 初始化视觉投影权重
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        # 如果模块是 LayerNorm，初始化层归一化的偏置和权重
        if isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 的偏置设置为 0
            module.bias.data.zero_()
            # 将 LayerNorm 的权重设置为 1
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Linear 并且含有偏置，初始化其偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将线性层的偏置设置为 0
            module.bias.data.zero_()
# OWLVIT_START_DOCSTRING为模型文档字符串的起始部分
OWLVIT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OwlViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# OWLVIT_TEXT_INPUTS_DOCSTRING为文本输入文档字符串的描述
OWLVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# OWLVIT_VISION_INPUTS_DOCSTRING为视觉输入文档字符串的描述
OWLVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# OWLVIT_INPUTS_DOCSTRING为输入文档字符串的描述
OWLVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。索引可以使用 [`AutoTokenizer`] 获得。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。 [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，避免对填充标记索引执行注意力操作。在 `[0, 1]` 中选择的掩码值：
            # - 对于**未遮罩**的标记，为 1，
            # - 对于**已遮罩**的标记，为 0。
            [什么是注意力遮罩?](../glossary#attention-mask)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
```  
# 定义 OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING 常量，描述输入参数的作用和形状
OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids).
        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the last hidden state. See `text_model_last_hidden_state` and
            `vision_model_last_hidden_state` under returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING 常量，描述输入参数的作用和形状
OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values of query image(s) to be detected. Pass in one query image per target image.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 OwlViTEncoder 类，表示包含 config.num_hidden_layers 自注意力层的Transformer编码器
class OwlViTEncoder(nn.Module):

    # 初始化方法，接收 OwlViTConfig 对象作为参数
    def __init__(self, config: OwlViTConfig):
        super().__init__()
        # 创建包含 config.num_hidden_layers 个 OwlViTEncoderLayer 的模块列表
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False
    # 定义前向传播函数
    def forward(
        self,
        # 输入的 Embedding 特征
        inputs_embeds,
        # 注意力掩码张量，用于指示哪些部分需要被注意力机制关注
        attention_mask: Optional[torch.Tensor] = None,
        # 因果注意力掩码张量，用于在生成文本时指定注意力机制的因果属性
        causal_attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回完整的结果字典
        return_dict: Optional[bool] = None,
# 创建 OwlViTTextTransformer 类，继承自 nn.Module
class OwlViTTextTransformer(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config: OwlViTTextConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的 config 参数赋值给实例变量
        self.config = config
        # 从配置中获取隐藏层的维度作为嵌入维度
        embed_dim = config.hidden_size
        # 创建 OwlViTTextEmbeddings 实例并赋值给实例变量
        self.embeddings = OwlViTTextEmbeddings(config)
        # 创建 OwlViTEncoder 实例并赋值给实例变量
        self.encoder = OwlViTEncoder(config)
        # 创建 LayerNorm 实例并赋值给实例变量，用于最终的归一化处理
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 使用装饰器将下面的 forward 函数添加文档字符串
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    # 使用装饰器将下面的 forward 函数的返回值类型替换为 BaseModelOutputWithPooling，配置类替换为 OwlViTTextConfig
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    # 定义前向传播函数，接受多个输入参数
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 返回文本编码器的输出
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            This method takes in various input tensors and configuration options and returns the output of the text encoder.
        """
        # 设置输出Attention和Hidden States的标志
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 获取输入的形状并展平
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # 通过embeddings获取输入的隐藏状态
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
    
        # 准备因果注意力掩码
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # 如果提供了attention_mask，则扩展它
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
    
        # 通过编码器获取输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取最后一个隐藏状态并进行归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
    
        # 获取每个序列的最后一个token的隐藏状态作为pooled_output
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]
    
        # 根据return_dict的值返回不同格式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个文本模型类，继承自OwlViTPreTrainedModel
class OwlViTTextModel(OwlViTPreTrainedModel):
    # 将配置类设置为OwlViTTextConfig
    config_class = OwlViTTextConfig

    # 初始化方法，接受一个配置参数config，调用父类的初始化方法
    def __init__(self, config: OwlViTTextConfig):
        super().__init__(config)
        # 创建一个文本转换器模型对象
        self.text_model = OwlViTTextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # 前向传播方法，接受多个参数，返回BaseModelOutputWithPooling类型的对象
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回结果：

        示例：
        ```py
        >>> from transformers import AutoProcessor, OwlViTTextModel

        >>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # 获取所有文本查询在所有批次样本中的嵌入
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 定义一个视觉变换器类，继承自nn.Module
class OwlViTVisionTransformer(nn.Module):
    # 初始化方法，接受一个配置参数config
    def __init__(self, config: OwlViTVisionConfig):
        super().__init__()
        self.config = config

        # 创建视觉嵌入层对象
        self.embeddings = OwlViTVisionEmbeddings(config)
        # 创建预处理的LayerNorm层对象
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建编码器对象
        self.encoder = OwlViTEncoder(config)
        # 创建后处理的LayerNorm层对象
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接受多个参数，返回BaseModelOutputWithPooling类型的对象
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回：
        """
        # 如果未提供输出注意力张量，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供输出隐藏状态张量，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供返回字典，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据类型转换为预期的数据类型
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        # 将像素值转换为嵌入向量
        hidden_states = self.embeddings(pixel_values)
        # 对嵌入向量进行预层归一化处理
        hidden_states = self.pre_layernorm(hidden_states)

        # 编码器处理嵌入向量，生成编码器输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 获取池化输出，取每个序列的第一个向量
        pooled_output = last_hidden_state[:, 0, :]

        # 对池化输出进行后层归一化处理
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带有池化输出和编码器输出的字典
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义 OwlViTVisionModel 类，继承自 OwlViTPreTrainedModel 类
class OwlViTVisionModel(OwlViTPreTrainedModel):
    # 指定配置类为 OwlViTVisionConfig
    config_class = OwlViTVisionConfig
    # 设置主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受一个 OwlViTVisionConfig 类型的参数
    def __init__(self, config: OwlViTVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 vision_model 属性为 OwlViTVisionTransformer 实例
        self.vision_model = OwlViTVisionTransformer(config)
        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法，接受多个参数并返回一个 BaseModelOutputWithPooling 实例或元组
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回 vision_model 的前向传播结果

        Examples:
        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OwlViTVisionModel

        >>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 为 OwlViTModel 类添加起始文档注释
@add_start_docstrings(OWLVIT_START_DOCSTRING)
# 定义 OwlViTModel 类，继承自 OwlViTPreTrainedModel 类
class OwlViTModel(OwlViTPreTrainedModel):
    # 指���配置类为 OwlViTConfig
    config_class = OwlViTConfig
    # 初始化函数，接受一个配置参数 OwlViTConfig
    def __init__(self, config: OwlViTConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置参数中的文本配置不是 OwlViTTextConfig 类型，则抛出数值错误
        if not isinstance(config.text_config, OwlViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type OwlViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 如果配置参数中的视觉配置不是 OwlViTVisionConfig 类型，则抛出数值错误
        if not isinstance(config.vision_config, OwlViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type OwlViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将文本配置和视觉配置存储到对应变量中
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)

        # 初始化视觉投影和文本投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型前向方法添加文档字符串
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # 返回文本特征，类型为 torch.FloatTensor
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTTextModel`].
            
        Examples:
        ```py
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        
        # 如果 return_dict 未指定则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取所有文本查询的嵌入
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        pooled_output = text_output[1]
        text_features = self.text_projection(pooled_output)

        return text_features
    # 为模型的前向传播添加文档字符串，描述输入参数的含义
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 像素数值，可选的浮点数张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选的布尔值，默认为 None
    ) -> torch.FloatTensor:  # 返回值类型为浮点数张量
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTVisionModel`].

        Examples:
        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # 如果指定了一些字段（如果指定的话），则使用 OWL-ViT 模型的配置，而不是使用 vision 和 text 组件的配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对视觉模型进行前向传播
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = vision_outputs[1]
        # 应用投影层到池化输出，获得图像特征
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征
        return image_features
class OwlViTBoxPredictionHead(nn.Module):
    # OwlViTBoxPredictionHead 类定义
    def __init__(self, config: OwlViTConfig, out_dim: int = 4):
        # 初始化函数
        super().__init__()

        # 获取视觉配置的隐藏大小
        width = config.vision_config.hidden_size
        # 第一个全连接层，输入和输出大小都是 width
        self.dense0 = nn.Linear(width, width)
        # 第二个全连接层，输入和输出大小都是 width
        self.dense1 = nn.Linear(width, width)
        # GELU 激活函数
        self.gelu = nn.GELU()
        # 最后一个全连接层，输入大小是 width，输出大小是 out_dim
        self.dense2 = nn.Linear(width, out_dim)

    # 前向传播函数
    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        # 第一个全连接层
        output = self.dense0(image_features)
        # GELU 激活函数
        output = self.gelu(output)
        # 第二个全连接层
        output = self.dense1(output)
        # GELU 激活函数
        output = self.gelu(output)
        # 最后一个全连接层
        output = self.dense2(output)
        # 返回输出
        return output


class OwlViTClassPredictionHead(nn.Module):
    # OwlViTClassPredictionHead 类定义
    def __init__(self, config: OwlViTConfig):
        # 初始化函数
        super().__init__()

        # 输出维度为文本配置的隐藏大小
        out_dim = config.text_config.hidden_size
        # 查询维度为视觉配置的隐藏大小
        self.query_dim = config.vision_config.hidden_size

        # 第一个全连接层，输入大小是查询维度，输出大小是输出维度
        self.dense0 = nn.Linear(self.query_dim, out_dim)
        # 可学习的偏移量
        self.logit_shift = nn.Linear(self.query_dim, 1)
        # 可学习的缩放因子
        self.logit_scale = nn.Linear(self.query_dim, 1)
        # ELU 激活函数
        self.elu = nn.ELU()

    # 前向传播函数
    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        # 图像类别嵌入
        image_class_embeds = self.dense0(image_embeds)
        # 如果没有查询嵌入
        if query_embeds is None:
            # 创建全零的预测对数张量
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            # 返回预测对数和图像类别嵌入
            return (pred_logits, image_class_embeds)

        # 归一化图像和文本特征
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # 获取类别预测对数
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # 应用可学习的偏移和缩放到对数中
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        # 如果有查询掩码
        if query_mask is not None:
            # 增加维度以匹配 logits
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            # 将未激活区域设置为极小值
            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        # 返回预测对数和图像类别嵌入
        return (pred_logits, image_class_embeds)


class OwlViTForObjectDetection(OwlViTPreTrainedModel):
    # OwlViTForObjectDetection 类定义
    config_class = OwlViTConfig
    # 初始化方法，接受一个OwlViTConfig对象作为参数
    def __init__(self, config: OwlViTConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建OwlViTModel对象
        self.owlvit = OwlViTModel(config)
        # 创建OwlViTClassPredictionHead对象
        self.class_head = OwlViTClassPredictionHead(config)
        # 创建OwlViTBoxPredictionHead对象
        self.box_head = OwlViTBoxPredictionHead(config)

        # 创建LayerNorm层，对特征进行归一化
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        # 创建Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 归一化网格角坐标
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # 计算特征图中的xy角坐标，并进行归一化
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        # 获取设备信息
        device = feature_map.device
        num_patches = feature_map.shape[1]

        # 创建网格的坐标，并进行归一化
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # 将 (h, w, 2) 形状的数组展平为 (h*w, 2) 形状的数组
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    # 计算边界框偏差
    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # 边界框中心偏差与特征网格上的位置相关
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # 对xy进行反归一化
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # 边界框大小与补丁大小相关
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # 计算边界框偏差
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    # 边界框预测器
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                从图像提��的特征，由'image_text_embedder'方法返回。
            feature_map:
                图像特征的空间重新排列，也由'image_text_embedder'方法返回。
        Returns:
            pred_boxes:
                嵌套在字典中的预测框列表（cxcywh归一化到0,1）。
        """
        # 边界框检测头 [batch_size, num_boxes, 4]。
        pred_boxes = self.box_head(image_feats)

        # 计算每个标记在网格上的位置，并用它来计算bbox预测的偏差
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes
    # 这个函数用于预测类别
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                从 image_text_embedder 中提取的图像特征。
            query_embeds:
                文本查询嵌入。
            query_mask:
                与 query_embeddings 一起提供。指示哪些查询嵌入是有效的。
        """
        # 将图像特征、查询嵌入和查询掩码传递给 class_head 层，获得预测的类别逻辑值和图像类别嵌入
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)
    
        # 返回预测的类别逻辑值和图像类别嵌入
        return (pred_logits, image_class_embeds)
    
    # 这个函数用于编码图像和文本
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # 编码文本和图像
        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    
        # 获取图像嵌入
        last_hidden_state = outputs.vision_model_output[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)
    
        # 调整类别tokens的大小
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
    
        # 将类别tokens与图像嵌入合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
    
        # 调整图像嵌入大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]
    
        # 返回文本嵌入、图像嵌入和其他输出
        return (text_embeds, image_embeds, outputs)
    
    # 这个函数用于编码图像
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # 获取 OwlViTModel 视觉嵌入（与 CLIP 相同）
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values, return_dict=True)

        # 对最后隐藏状态应用后层归一化，返回非投影输出
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # 调整类别标记大小
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # 将图像嵌入与类别标记合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    def embed_image_query(
        self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 使用查询图像特征进行类别预测，返回类别预测结果和类别嵌入
        _, class_embeds = self.class_predictor(query_image_features)
        # 使用查询图像特征进行边界框预测，返回预测的边界框
        pred_boxes = self.box_predictor(query_image_features, query_feature_map)
        # 将预测的边界框转换为以边角表示的格式
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)

        # 遍历查询图像
        best_class_embeds = []
        best_box_indices = []
        # 获取预测边界框所在设备
        pred_boxes_device = pred_boxes_as_corners.device

        for i in range(query_image_features.shape[0]):
            # 创建每个查询边界框，初始值为[[0, 0, 1, 1]]
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            # 计算每个查询边界框与预测边界框的IoU
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # 如果没有重叠的边界框，使用广义IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # 使用自适应阈值，包含所有在最佳IoU的80%范围内的边界框
            iou_threshold = torch.max(ious) * 0.8

            # 选择符合条件的边界框索引
            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        # 如果有最佳类别嵌入，将其堆叠成张量
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        # 返回查询嵌入、边界框索引和预测边界框
        return query_embeds, box_indices, pred_boxes

    @add_start_docstrings_to_model_forward(OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTImageGuidedObjectDetectionOutput, config_class=OwlViTConfig)
    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    @add_start_docstrings_to_model_forward(OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTObjectDetectionOutput, config_class=OwlViTConfig)
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```