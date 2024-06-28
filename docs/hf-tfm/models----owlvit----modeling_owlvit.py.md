# `.\models\owlvit\modeling_owlvit.py`

```py
# 设置代码文件的字符编码为UTF-8
# Copyright声明和许可证信息，使用Apache License 2.0
# 详情参见：http://www.apache.org/licenses/LICENSE-2.0
# 本段代码是PyTorch版本的OWL-ViT模型定义

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

# 导入OWL-ViT模型中的激活函数映射
from ...activations import ACT2FN
# 导入OWL-ViT模型中的注意力掩码相关的函数
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
# 导入OWL-ViT模型的输出类
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
# 导入OWL-ViT模型的基类PreTrainedModel
from ...modeling_utils import PreTrainedModel
# 导入通用的工具函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)
# 导入OWL-ViT配置类
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig

# 如果视觉处理可用，则导入视觉相关的转换函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点路径
_CHECKPOINT_FOR_DOC = "google/owlvit-base-patch32"

# 可用的预训练模型列表，参见 https://huggingface.co/models?filter=owlvit
OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlvit-base-patch32",
    "google/owlvit-base-patch16",
    "google/owlvit-large-patch14",
]

# 定义对比损失函数，用于OWL-ViT模型
# 从transformers.models.clip.modeling_clip.contrastive_loss复制并修改为owlvit
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 定义OWL-ViT模型特定的损失函数
# 从transformers.models.clip.modeling_clip.clip_loss复制并修改为owlvit
def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

# 定义OWL-ViT模型的输出类，继承自ModelOutput
@dataclass
class OwlViTOutput(ModelOutput):
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`OwlViTVisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """

    # Optional attribute: Contrastive loss for image-text similarity
    loss: Optional[torch.FloatTensor] = None
    # Tensor: Scores of image-text similarity, shape (image_batch_size, text_batch_size)
    logits_per_image: torch.FloatTensor = None
    # Tensor: Scores of text-image similarity, shape (text_batch_size, image_batch_size)
    logits_per_text: torch.FloatTensor = None
    # Tensor: Embeddings of text, shape (batch_size * num_max_text_queries, output_dim)
    text_embeds: torch.FloatTensor = None
    # Tensor: Embeddings of images, shape (batch_size, output_dim)
    image_embeds: torch.FloatTensor = None
    # Tuple[`BaseModelOutputWithPooling`]: Output of text model
    text_model_output: BaseModelOutputWithPooling = None
    # `BaseModelOutputWithPooling`: Output of vision model

    def to_tuple(self) -> Tuple[Any]:
        # Convert object attributes to a tuple, handling special cases for complex attributes
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# Copied from transformers.models.detr.modeling_detr._upcast
# 将输入的张量升级到更高的数据类型，以防止在乘法操作中出现数值溢出
def _upcast(t: Tensor) -> Tensor:
    if t.is_floating_point():
        # 如果输入张量已经是浮点类型，则直接返回
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果输入张量是整型，则将其升级为对应的整型类型
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
# 计算一组边界框的面积，边界框通过其 (x1, y1, x2, y2) 坐标来指定
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，边界框的格式为 (x1, y1, x2, y2)，其中 `0 <= x1 < x2` 且 `0 <= y1 < y2`。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            待计算面积的边界框。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
# 计算两组边界框之间的 IoU（Intersection over Union）
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 计算交集的左上角和右下角坐标
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集区域的宽度和高度
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    # 计算并集的面积
    union = area1[:, None] + area2 - inter

    # 计算 IoU
    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
# 计算广义 IoU，支持包括不完全矩形在内的任意形状的边界框
def generalized_box_iou(boxes1, boxes2):
    """
    根据 https://giou.stanford.edu/ 计算广义 IoU。边界框应为 [x0, y0, x1, y1]（左上角和右下角）格式。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 检查是否存在退化的边界框，这些边界框会导致 inf / nan 的结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须以 [x0, y0, x1, y1]（左上角和右下角）格式给出，但得到的是 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须以 [x0, y0, x1, y1]（左上角和右下角）格式给出，但得到的是 {boxes2}")
    
    # 计算标准 IoU 和并集面积
    iou, union = box_iou(boxes1, boxes2)

    # 计算最小外接矩形的左上角和右下角坐标
    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算最小外接矩形的宽度和高度
    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    # 计算广义 IoU
    return iou - (area - union) / area


@dataclass
class OwlViTObjectDetectionOutput(ModelOutput):
    """
    [`OwlViTForObjectDetection`] 的输出类型。
    """
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

    # Optional attributes initialized to `None` by default
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # Method to convert the attributes to a tuple, excluding specific complex types
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
@dataclass
class OwlViTImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
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

    # 定义输出类，用于图像引导物体检测的结果
    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 转换为元组的方法，将类的字段转换为元组，除了"text_model_output"和"vision_model_output"外，它们将被转换为元组形式
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    # 初始化函数，接受一个 OwlViTVisionConfig 类型的配置对象作为参数
    def __init__(self, config: OwlViTVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存在实例变量中
        self.config = config
        # 设置嵌入维度为配置对象中的隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化类嵌入，使用随机生成的张量作为参数
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # 创建图块嵌入层
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,          # 输入通道数为配置对象中的通道数
            out_channels=self.embed_dim,              # 输出通道数为嵌入维度
            kernel_size=config.patch_size,            # 卷积核大小为配置对象中的图块大小
            stride=config.patch_size,                 # 步幅为配置对象中的图块大小
            bias=False,                              # 不使用偏置项
        )

        # 计算图块数目，即图像尺寸除以图块大小的平方
        self.num_patches = (config.image_size // config.patch_size) ** 2
        # 设置位置嵌入层的位置数目，等于图块数目加一
        self.num_positions = self.num_patches + 1
        # 创建位置嵌入层，使用 Embedding 类，位置数目为 num_positions，嵌入维度为 embed_dim
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册位置 ID 缓冲区，创建一个张量表示从 0 到 num_positions-1 的序列
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播函数，接受像素值作为输入，返回嵌入向量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]
        # 对输入像素值进行图块嵌入，输出形状为 [batch_size, num_channels, height, width]
        patch_embeds = self.patch_embedding(pixel_values)
        # 将图块嵌入展平并转置，形状为 [batch_size, num_patches, embed_dim]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类嵌入，形状为 [batch_size, 1, embed_dim]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 拼接类嵌入和图块嵌入，形状为 [batch_size, num_patches + 1, embed_dim]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上位置嵌入，形状为 [batch_size, num_patches + 1, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回嵌入向量
        return embeddings
class OwlViTTextEmbeddings(nn.Module):
    # OwlViTTextEmbeddings 类，用于处理文本嵌入
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()
        # 初始化 token_embedding 层，用于词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # 初始化 position_embedding 层，用于位置编码
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 创建 position_ids 张量，并注册为缓冲区，用于处理位置编码
        # 这个张量在序列化时会被导出，位置是内存中的连续存储
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果未提供 position_ids，则使用预先创建的 position_ids 张量
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 inputs_embeds，则通过 token_embedding 层获取
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置编码的嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 计算最终的嵌入表示，包括词嵌入和位置编码的和
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class OwlViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 初始化线性投影层，用于查询（q_proj）、键（k_proj）、值（v_proj）和输出（out_proj）
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新形状张量，以便进行多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        # 省略了具体的前向传播代码，但通常会涉及到自注意力机制和线性投影操作
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->OwlViT
class OwlViTMLP(nn.Module):
    # OwlViTMLP 类，用于多层感知机（MLP）部分的实现
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 激活函数根据配置选择
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个全连接层，将隐藏大小映射到中间大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个全连接层，将中间大小映射回隐藏大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    # 定义前向传播方法，接收隐藏状态张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量通过全连接层 fc1
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数到 fc1 的输出上
        hidden_states = self.activation_fn(hidden_states)
        # 将激活后的隐藏状态张量通过全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的张量作为前向传播的结果
        return hidden_states
# 从 transformers.models.clip.modeling_clip.CLIPEncoderLayer 复制而来，修改为 OwlViTEncoderLayer
class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏大小
        self.self_attn = OwlViTAttention(config)  # 初始化自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化层
        self.mlp = OwlViTMLP(config)  # 多层感知机模块
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化层

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        前向传播函数
        Args:
            hidden_states (`torch.FloatTensor`): 形状为 `(batch, seq_len, embed_dim)` 的输入状态
            attention_mask (`torch.FloatTensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，
                其中填充元素由非常大的负值表示。
            causal_attention_mask (`torch.FloatTensor`): 大小为 `(config.encoder_attention_heads,)` 的因果注意力掩码。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。详细信息请参阅返回张量中的 `attentions`。
        """
        residual = hidden_states  # 保留残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 应用第一个归一化层
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )  # 应用自注意力机制，获取注意力权重
        hidden_states = residual + hidden_states  # 添加残差连接

        residual = hidden_states  # 更新残差连接变量
        hidden_states = self.layer_norm2(hidden_states)  # 应用第二个归一化层
        hidden_states = self.mlp(hidden_states)  # 应用多层感知机模块
        hidden_states = residual + hidden_states  # 添加残差连接

        outputs = (hidden_states,)  # 设置输出为隐藏状态张量

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出元组中

        return outputs  # 返回输出元组作为前向传播的结果


class OwlViTPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化、下载预训练模型及简单接口的抽象类。
    """

    config_class = OwlViTConfig  # 使用 OwlViTConfig 类来配置模型
    base_model_prefix = "owlvit"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["OwlViTEncoderLayer"]  # 不进行模块分割的模块列表
    # 初始化模型权重函数，用于为给定模块初始化权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 从配置中获取初始化因子
        factor = self.config.initializer_factor
        
        # 如果模块是 OwlViTTextEmbeddings 类型，则初始化其权重
        if isinstance(module, OwlViTTextEmbeddings):
            # 初始化 token_embedding 和 position_embedding 的权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        
        # 如果模块是 OwlViTVisionEmbeddings 类型，则初始化其权重
        elif isinstance(module, OwlViTVisionEmbeddings):
            # 初始化 class_embedding、patch_embedding 和 position_embedding 的权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        
        # 如果模块是 OwlViTAttention 类型，则初始化其权重
        elif isinstance(module, OwlViTAttention):
            # 初始化 attention 层的权重：q_proj、k_proj、v_proj 和 out_proj
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        
        # 如果模块是 OwlViTMLP 类型，则初始化其权重
        elif isinstance(module, OwlViTMLP):
            # 初始化 MLP 层的权重：fc1 和 fc2
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        
        # 如果模块是 OwlViTModel 类型，则初始化其权重
        elif isinstance(module, OwlViTModel):
            # 初始化模型的 text_projection 和 visual_projection 的权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        
        # 如果模块是 nn.LayerNorm 类型，则对其进行归一化操作
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # 如果模块是 nn.Linear 类型且具有偏置，则将偏置数据初始化为零
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# 定义 OWLVIT 模型输入文档字符串，说明模型的输入参数及其形状
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

# 定义 OWLVIT 文本输入文档字符串，描述文本输入相关参数
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

# 定义 OWLVIT 视觉输入文档字符串，描述视觉输入相关参数
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

# 定义 OWLVIT 输入文档字符串，作为 OWLVIT_TEXT_INPUTS_DOCSTRING 和 OWLVIT_VISION_INPUTS_DOCSTRING 的总结
OWLVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的标记索引，形状为(batch_size, sequence_length)，可以使用AutoTokenizer获取。参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using `AutoTokenizer`. See
            `PreTrainedTokenizer.encode` and `PreTrainedTokenizer.__call__` for details.
            [What are input IDs?](../glossary#input-ids)
        
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 注意力掩码，形状为(batch_size, sequence_length)，用于避免在填充的标记索引上执行注意力操作。掩码值选择在[0, 1]：
            # - 1 表示**未掩码**的标记，
            # - 0 表示**掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值，形状为(batch_size, num_channels, height, width)的浮点张量。
            Pixel values.
        
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
            Whether or not to return the contrastive loss.
        
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量中的`attentions`以获取更多详细信息。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量中的`hidden_states`以获取更多详细信息。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        return_dict (`bool`, *optional*):
            # 是否返回`utils.ModelOutput`而不是普通的元组。
            Whether or not to return a `utils.ModelOutput` instead of a plain tuple.
"""

# OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING 是一个原始字符串，用于描述 OwlViT 模型输入的文档字符串。
"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。
    input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*):
        输入序列标记在词汇表中的索引。可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。[输入 ID 是什么？](../glossary#input-ids)。
    attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
        遮罩，避免对填充的标记索引执行注意力操作。遮罩值在 `[0, 1]` 之间：
        - 1 表示**未被遮罩**的标记，
        - 0 表示**被遮罩**的标记。
        [注意力遮罩是什么？](../glossary#attention-mask)
    output_hidden_states (`bool`, *optional*):
        是否返回最后一个隐藏状态。详见返回张量中的 `text_model_last_hidden_state` 和 `vision_model_last_hidden_state`。
    return_dict (`bool`, *optional*):
        是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING 是一个原始字符串，用于描述 OwlViT 图像导向目标检测模型输入的文档字符串。
"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。
    query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        待检测的查询图像的像素值。每个目标图像传入一个查询图像。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions`。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states`。
    return_dict (`bool`, *optional*):
        是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

class OwlViTEncoder(nn.Module):
    """
    OwlViT 编码器，包含 `config.num_hidden_layers` 个自注意力层。每层都是一个 [`OwlViTEncoderLayer`]。

    Args:
        config: OwlViTConfig
    """

    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 定义一个方法 `forward`，用于执行模型的前向传播操作
    def forward(
        # 输入的嵌入表示，通常是一个张量
        self,
        inputs_embeds,
        # 注意力掩码，可选的张量，默认为None，用于指定哪些位置的输入需要被忽略
        attention_mask: Optional[torch.Tensor] = None,
        # 因果注意力掩码，可选的张量，默认为None，用于自回归任务中避免信息泄漏
        causal_attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重的标志，可选的布尔值，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态的标志，可选的布尔值，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回输出结果的标志，可选的布尔值，默认为None
        return_dict: Optional[bool] = None,
# 定义一个名为 OwlViTTextTransformer 的类，继承自 nn.Module
class OwlViTTextTransformer(nn.Module):
    
    # 初始化方法，接受一个 OwlViTTextConfig 类型的参数 config
    def __init__(self, config: OwlViTTextConfig):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量 self.config 中
        self.config = config
        # 根据配置中的 hidden_size 设置 embed_dim 变量
        embed_dim = config.hidden_size
        # 创建 OwlViTTextEmbeddings 类的实例，并保存到实例变量 self.embeddings 中
        self.embeddings = OwlViTTextEmbeddings(config)
        # 创建 OwlViTEncoder 类的实例，并保存到实例变量 self.encoder 中
        self.encoder = OwlViTEncoder(config)
        # 创建一个 LayerNorm 层，用于对最终输出进行归一化，指定归一化的维度为 embed_dim
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 前向传播方法，通过装饰器添加了输入文档和返回文档的注释
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
        self,
        input_ids: torch.Tensor,                           # 输入的 token IDs 张量
        attention_mask: Optional[torch.Tensor] = None,     # 注意力掩码张量，可选
        position_ids: Optional[torch.Tensor] = None,       # 位置 IDs 张量，可选
        output_attentions: Optional[bool] = None,          # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,       # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,                # 是否以字典形式返回结果，可选
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        # 如果没有显式提供 output_attentions 参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式提供 output_hidden_states 参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式提供 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入张量的形状
        input_shape = input_ids.size()
        # 将 input_ids 转换为二维张量，去除 batch 维度
        input_ids = input_ids.view(-1, input_shape[-1])
        # 使用 embeddings 层处理 input_ids 和 position_ids
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 根据输入形状和隐藏状态数据类型，创建 causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # 如果提供了 attention_mask，则将其扩展到四维张量
        if attention_mask is not None:
            # 将二维 attention_mask 扩展为四维张量
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 使用 encoder 处理隐藏状态，传递参数包括输入嵌入、注意力掩码等
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 encoder 输出的最后隐藏状态并进行 layer normalization
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 从 tokens 嵌入的末尾获取特征（每个序列中的最大数值处）
        # 为了兼容 ONNX，将 input_ids 转换为整型再执行 argmax 操作
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        # 如果不使用 return_dict，则返回一个元组，包括最后隐藏状态、汇总输出和额外的 encoder 输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用 BaseModelOutputWithPooling 创建返回字典，包括最后隐藏状态、汇总输出、隐藏状态和注意力矩阵
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# OwlViTTextModel 类，继承自 OwlViTPreTrainedModel 类
class OwlViTTextModel(OwlViTPreTrainedModel):
    # 指定配置类为 OwlViTTextConfig
    config_class = OwlViTTextConfig

    # 初始化方法
    def __init__(self, config: OwlViTTextConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 OwlViTTextTransformer 对象作为文本模型
        self.text_model = OwlViTTextTransformer(config)
        # 执行初始化权重和最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的 token 嵌入层
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        # 设置文本模型的 token 嵌入层
        self.text_model.embeddings.token_embedding = value

    # 前向传播方法，使用装饰器添加文档字符串
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
        Returns:

        Examples:
        ```
        >>> from transformers import AutoProcessor, OwlViTTextModel

        >>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astronaut"]], return_tensors="pt"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # 调用文本模型的前向传播方法，获取所有批次样本的文本查询嵌入
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# OwlViTVisionTransformer 类，继承自 nn.Module 类
class OwlViTVisionTransformer(nn.Module):
    # 初始化方法
    def __init__(self, config: OwlViTVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 存储配置信息
        self.config = config

        # 创建视觉嵌入层对象
        self.embeddings = OwlViTVisionEmbeddings(config)
        # 创建预层归一化层对象
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建编码器对象
        self.encoder = OwlViTEncoder(config)
        # 创建后层归一化层对象
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，使用装饰器添加文档字符串
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 此处省略部分前向传播代码
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回：
        返回类型注解声明函数返回的类型为元组或BaseModelOutputWithPooling类的实例。

        """
        # 确定是否输出注意力权重，默认为模型配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认为模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定返回类型是否为字典形式，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据转换为预期的数据类型
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        # 通过嵌入层处理像素值
        hidden_states = self.embeddings(pixel_values)
        # 应用预层归一化处理隐藏状态
        hidden_states = self.pre_layernorm(hidden_states)

        # 将隐藏状态输入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取池化输出，通常是最后隐藏状态的第一个位置
        pooled_output = last_hidden_state[:, 0, :]

        # 对池化输出进行后层归一化处理
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不需要返回字典形式的输出，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典形式的输出，则创建BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个视觉模型类，继承自预训练模型 OwlViTPreTrainedModel
class OwlViTVisionModel(OwlViTPreTrainedModel):
    # 指定配置类为 OwlViTVisionConfig
    config_class = OwlViTVisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接收配置参数 config
    def __init__(self, config: OwlViTVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建视觉模型实例，使用 OwlViTVisionTransformer 类来构建
        self.vision_model = OwlViTVisionTransformer(config)
        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入的方法，返回视觉模型的嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法装饰器，添加模型输入文档字符串和输出文档字符串
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
        前向传播函数，接收像素值、是否输出注意力、是否输出隐藏状态和是否返回字典作为参数。

        返回:
        - BaseModelOutputWithPooling 或者 Tuple
            模型的输出，可能包括最后的隐藏状态和汇聚输出（汇聚的CLS状态）。

        示例:
        ```
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
        >>> pooled_output = outputs.pooler_output  # 汇聚的CLS状态
        ```
        """
        # 调用视觉模型的前向传播方法，传递所有参数
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 使用起始文档字符串装饰器添加 OWLVIT_START_DOCSTRING 的类 OwlViTModel
@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OwlViTModel(OwlViTPreTrainedModel):
    # 指定配置类为 OwlViTConfig
    config_class = OwlViTConfig
    def __init__(self, config: OwlViTConfig):
        super().__init__(config)

        # 检查并确保config.text_config是OwlViTTextConfig类型
        if not isinstance(config.text_config, OwlViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type OwlViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查并确保config.vision_config是OwlViTVisionConfig类型
        if not isinstance(config.vision_config, OwlViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type OwlViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从config中获取text_config和vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化模型的参数
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化text_model和vision_model
        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)

        # 初始化visual_projection和text_projection线性层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 初始化logit_scale参数
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # 初始化完成后执行后处理函数
        self.post_init()

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTTextModel`].

        Examples:
        ```
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astronaut"]], return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 根据self.config.use_return_dict的设置决定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用text_model获取文本输出
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        # 从文本输出中获取汇总的输出
        pooled_output = text_output[1]
        # 将汇总的输出通过text_projection线性层进行映射，得到文本特征
        text_features = self.text_projection(pooled_output)

        return text_features
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTVisionModel`].

        Examples:
        ```
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

        # Use OWL-ViT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the input pixel values and optional flags to the vision model
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the pooled output from vision model outputs
        pooled_output = vision_outputs[1]
        
        # Apply projection layer to the pooled output to obtain image features
        image_features = self.visual_projection(pooled_output)

        # Return the computed image features
        return image_features

    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_base_image_embeds: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        This method performs the forward pass of the OwlViT model.

        Returns:
            output (:class:`~transformers.OwlViTOutput` or :obj:`torch.FloatTensor`):
                The output of the OwlViT model, containing various elements depending on the configuration.
        """

        # Set default values for optional parameters if not provided explicitly
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the vision and text integration model with specified inputs and configuration
        model_outputs = self.owlvit_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_base_image_embeds=return_base_image_embeds,
            return_dict=return_dict,
        )

        # Return the model outputs
        return model_outputs
class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, out_dim: int = 4):
        super().__init__()

        # 获取视觉配置中的隐藏层大小作为宽度
        width = config.vision_config.hidden_size
        # 创建线性层 dense0，输入和输出大小都为 width
        self.dense0 = nn.Linear(width, width)
        # 创建线性层 dense1，输入和输出大小都为 width
        self.dense1 = nn.Linear(width, width)
        # 创建 GELU 激活函数
        self.gelu = nn.GELU()
        # 创建线性层 dense2，输入为 width，输出为 out_dim
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        # 将输入特征 image_features 经过 dense0 线性层
        output = self.dense0(image_features)
        # 应用 GELU 激活函数
        output = self.gelu(output)
        # 将结果经过 dense1 线性层
        output = self.dense1(output)
        # 再次应用 GELU 激活函数
        output = self.gelu(output)
        # 将最终结果经过 dense2 线性层
        output = self.dense2(output)
        return output


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        # 根据配置获取文本配置中的隐藏层大小作为输出维度
        out_dim = config.text_config.hidden_size
        # 获取视觉配置中的隐藏层大小作为查询维度
        self.query_dim = config.vision_config.hidden_size

        # 创建线性层 dense0，输入大小为 self.query_dim，输出大小为 out_dim
        self.dense0 = nn.Linear(self.query_dim, out_dim)
        # 创建线性层 logit_shift，输入大小为 self.query_dim，输出大小为 1
        self.logit_shift = nn.Linear(self.query_dim, 1)
        # 创建线性层 logit_scale，输入大小为 self.query_dim，输出大小为 1
        self.logit_scale = nn.Linear(self.query_dim, 1)
        # 创建 ELU 激活函数
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        # 将图像嵌入经过 dense0 线性层
        image_class_embeds = self.dense0(image_embeds)
        
        # 如果查询嵌入为空，则创建全零预测 logits
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # 对图像和文本特征进行归一化
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # 获取类别预测 logits
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # 对 logits 应用可学习的偏移和缩放
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        # 如果存在查询掩码，则应用掩码
        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)


class OwlViTForObjectDetection(OwlViTPreTrainedModel):
    config_class = OwlViTConfig
    def __init__(self, config: OwlViTConfig):
        # 调用父类构造函数初始化
        super().__init__(config)

        # 初始化 OwlViT 模型
        self.owlvit = OwlViTModel(config)
        # 初始化 OwlViT 分类头部
        self.class_head = OwlViTClassPredictionHead(config)
        # 初始化 OwlViT 边界框头部
        self.box_head = OwlViTBoxPredictionHead(config)

        # 初始化 LayerNorm 层，用于归一化视觉特征的隐藏大小
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        # 初始化 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

        # 计算平方根数目的图块
        self.sqrt_num_patches = config.vision_config.image_size // config.vision_config.patch_size

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # 从特征图计算归一化的 xy 角落坐标
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        device = feature_map.device
        num_patches = feature_map.shape[1]

        # 使用 numpy 创建网格坐标，然后堆叠成二维数组
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # 将二维数组展平成一维数组
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        # 转换为 Torch 张量并放置在合适的设备上
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # 边界框中心相对于特征网格位置的偏差
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        # 将坐标限制在 [0.0, 1.0] 范围内
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # 对 xy 进行反归一化
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # 边界框大小相对于图块大小的偏差
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # 计算边界框偏差
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                从图像提取的特征，由 `image_text_embedder` 方法返回。
            feature_map:
                图像特征的空间重排列，也由 `image_text_embedder` 方法返回。
        Returns:
            pred_boxes:
                预测框的列表（归一化为 0 到 1 的 cxcywh 格式），嵌套在一个字典中。
        """
        # 边界框检测头 [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # 计算每个标记在网格上的位置，并用它来计算边界框预测的偏置
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                从 `image_text_embedder` 提取的图像特征。
            query_embeds:
                文本查询的嵌入向量。
            query_mask:
                必须与查询嵌入一起提供。指示哪些查询嵌入是有效的掩码。
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

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

        # 调整类标记的大小
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # 将图像嵌入与类标记合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]

        return (text_embeds, image_embeds, outputs)
    # 定义一个方法用于嵌入图像特征到 OwlViT 模型中
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # 调用 OwlViT 模型的视觉嵌入方法，并返回一个字典形式的输出
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values, return_dict=True)

        # 获取最后一个隐藏状态并应用 post_layernorm，返回非投影输出
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # 调整类别标记的大小
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # 将图像嵌入与类别标记合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # 调整形状为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        # 返回嵌入的图像和 OwlViT 模型的全部输出
        return (image_embeds, vision_outputs)

    # 定义一个方法用于嵌入图像查询
    def embed_image_query(
        self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor
    ):
    ) -> torch.FloatTensor:
        _, class_embeds = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features, query_feature_map)
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device

        for i in range(query_image_features.shape[0]):
            # Create a single query box for comparison
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            # Calculate IoU between the query box and predicted boxes
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, use generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to select relevant boxes based on IoU
            iou_threshold = torch.max(ious) * 0.8

            # Select indices of predicted boxes with IoU above the threshold
            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                # Select corresponding embeddings for selected boxes
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                # Compute mean similarity between mean embedding and selected embeddings
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                # Find the index of the box with the highest similarity
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        # Stack selected embeddings and box indices if any valid boxes are found
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

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
    ):
        # Implementation details of the image-guided object detection method
        pass

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
    ):
        # Forward method implementation for OwlViT model for object detection tasks
        pass
```