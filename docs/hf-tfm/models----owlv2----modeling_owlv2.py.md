# `.\models\owlv2\modeling_owlv2.py`

```
# 设置文件编码为 UTF-8
# 版权声明，标识该文件版权归 Google AI 和 The HuggingFace Team 所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则禁止使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，不提供任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证

""" PyTorch OWLv2 模型."""

# 引入警告模块
import warnings
# 引入数据类模块
from dataclasses import dataclass
# 引入类型提示
from typing import Any, Dict, Optional, Tuple, Union

# 引入 NumPy 库
import numpy as np
# 引入 PyTorch 库
import torch
# 引入 PyTorch 的检查点模块
import torch.utils.checkpoint
# 引入 PyTorch 的张量和神经网络模块
from torch import Tensor, nn

# 引入自定义的激活函数映射
from ...activations import ACT2FN
# 引入自定义的注意力掩码工具函数
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
# 引入自定义的模型输出类
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
# 引入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 引入工具类函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)
# 引入 OWLv2 配置类
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig

# 如果视觉可用，引入转换函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点地址
_CHECKPOINT_FOR_DOC = "google/owlv2-base-patch16-ensemble"

# OWLv2 预训练模型存档列表
# 查看所有 OWLv2 模型：https://huggingface.co/models?filter=owlv2
OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlv2-base-patch16-ensemble",
    # 查看所有 OWLv2 模型：https://huggingface.co/models?filter=owlv2
]

# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制的对比损失函数，替换为 owlv2
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 从 transformers.models.clip.modeling_clip.clip_loss 复制的 OWLv2 损失函数
def owlv2_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

# OWLv2 模型的输出数据类，继承自 ModelOutput
@dataclass
class Owlv2Output(ModelOutput):
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
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`Owlv2VisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    # Optional attribute: contrastive loss for image-text similarity
    loss: Optional[torch.FloatTensor] = None
    # Tensor attribute: scores of image-text similarity
    logits_per_image: torch.FloatTensor = None
    # Tensor attribute: scores of text-image similarity
    logits_per_text: torch.FloatTensor = None
    # Tensor attribute: embeddings of text data
    text_embeds: torch.FloatTensor = None
    # Tensor attribute: embeddings of image data
    image_embeds: torch.FloatTensor = None
    # Tuple attribute: output from text model with pooling
    text_model_output: BaseModelOutputWithPooling = None
    # Object attribute: output from vision model with pooling
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        # Convert all attributes to a tuple, handling special cases for complex objects
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # 如果输入张量是浮点型，则根据需要提升到更高的浮点类型，以避免数值溢出问题
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果输入张量是整型，则根据需要提升到更高的整型类型
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其（x1，y1，x2，y2）坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应该以（x1，y1，x2，y2）格式给出，其中 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    # 将边界框张量提升到相应的数值类型，以处理数值精度问题
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
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


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    使用 https://giou.stanford.edu/ 中定义的广义 IoU 计算方法。边界框应该以 [x0, y0, x1, y1]（角点）格式给出。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 如果存在退化的边界框，会导致无限大或非数值结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须以 [x0, y0, x1, y1]（角点）格式给出，但实际得到 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须以 [x0, y0, x1, y1]（角点）格式给出，但实际得到 {boxes2}")
    
    # 计算边界框的 IoU 和并集面积
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


@dataclass
class Owlv2ObjectDetectionOutput(ModelOutput):
    """
    [`Owlv2ForObjectDetection`] 的输出类型。
    """
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
        objectness_logits (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`):
            The objectness logits of all image patches. OWL-ViT represents images as a set of image patches where the
            total number of patches is (image_size / patch_size)**2.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes image
            embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    # Optional: Total loss combining cross-entropy and bounding box loss
    loss: Optional[torch.FloatTensor] = None
    # Optional: Dictionary containing individual losses
    loss_dict: Optional[Dict] = None
    # Optional: Classification logits (including no-object) for all queries
    logits: torch.FloatTensor = None
    # Optional: Objectness logits for all image patches
    objectness_logits: torch.FloatTensor = None
    # Optional: Normalized bounding box coordinates for all queries
    pred_boxes: torch.FloatTensor = None
    # Optional: Text embeddings obtained from Owlv2TextModel
    text_embeds: torch.FloatTensor = None
    # Optional: Image embeddings obtained from Owlv2VisionModel
    image_embeds: torch.FloatTensor = None
    # Optional: Class embeddings of all image patches
    class_embeds: torch.FloatTensor = None
    # Optional: Output of Owlv2TextModel, including pooling
    text_model_output: BaseModelOutputWithPooling = None
    # Optional: Output of Owlv2VisionModel, including pooling
    vision_model_output: BaseModelOutputWithPooling = None
    # 将对象转换为元组的方法定义，返回一个元组
    def to_tuple(self) -> Tuple[Any]:
        # 使用生成器表达式生成元组的每个元素
        return tuple(
            # 如果键不是"text_model_output"或"vision_model_output"，则取出该键对应的值
            self[k] if k not in ["text_model_output", "vision_model_output"] 
            # 否则，调用对象自身的属性 k 的 to_tuple 方法来获取值，并作为元素
            else getattr(self, k).to_tuple()
            # 遍历对象自身的所有键
            for k in self.keys()
        )
@dataclass
# 数据类装饰器，用于定义具有预定义字段的类
class Owlv2ImageGuidedObjectDetectionOutput(ModelOutput):
    """
    [`Owlv2ForObjectDetection.image_guided_detection`] 的输出类型。

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            所有查询的分类 logits（包括无对象）。
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            所有查询的标准化框坐标，表示为 (center_x, center_y, width, height)。这些值在 [0, 1] 范围内，
            相对于批次中每个目标图像的大小（忽略可能的填充）。您可以使用 [`~Owlv2ImageProcessor.post_process_object_detection`]
            来获取未标准化的边界框。
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            所有查询的标准化框坐标，表示为 (center_x, center_y, width, height)。这些值在 [0, 1] 范围内，
            相对于批次中每个查询图像的大小（忽略可能的填充）。您可以使用 [`~Owlv2ImageProcessor.post_process_object_detection`]
            来获取未标准化的边界框。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim)`):
            [`Owlv2VisionModel`] 的汇聚输出。OWLv2 将图像表示为一组图像补丁，并为每个补丁计算图像嵌入。
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim)`):
            [`Owlv2VisionModel`] 的汇聚输出。OWLv2 将图像表示为一组图像补丁，并为每个补丁计算图像嵌入。
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            所有图像补丁的类嵌入。OWLv2 将图像表示为一组图像补丁，其中补丁总数为 (image_size / patch_size)**2。
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            [`Owlv2TextModel`] 的输出。
        vision_model_output (`BaseModelOutputWithPooling`):
            [`Owlv2VisionModel`] 的输出。
    """

    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    # 定义一个方法将对象转换为元组形式，返回元组
    def to_tuple(self) -> Tuple[Any]:
        # 使用生成器表达式生成元组，遍历对象的键
        return tuple(
            # 如果键不在指定的列表中，则返回该键对应的值
            self[k] if k not in ["text_model_output", "vision_model_output"]
            # 否则，调用对象的相应属性的 to_tuple 方法并返回结果
            else getattr(self, k).to_tuple()
            for k in self.keys()  # 遍历对象的所有键
        )
# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTVisionEmbeddings with OwlViT->Owlv2
class Owlv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Owlv2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # Define patch embedding layer for image patches
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Calculate total number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # Positional embeddings for patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]

        # Extract patch embeddings from input image
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # flatten patches and transpose for attention

        # Expand class embeddings to match batch size
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        # Concatenate class embeddings with patch embeddings
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # Add positional embeddings to the combined embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTTextEmbeddings with OwlViT->Owlv2
class Owlv2TextEmbeddings(nn.Module):
    def __init__(self, config: Owlv2TextConfig):
        super().__init__()

        # Token embeddings based on vocabulary size and hidden size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embeddings based on maximum position embeddings and hidden size
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
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

        # If position_ids is not provided, use default position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # If inputs_embeds is not provided, compute token embeddings from input_ids
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # Get positional embeddings based on position_ids
        position_embeddings = self.position_embedding(position_ids)

        # Combine token embeddings with positional embeddings
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTAttention with OwlViT->Owlv2
class Owlv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存为实例变量
        self.config = config
        # 从配置中获取隐藏层大小，作为嵌入维度
        self.embed_dim = config.hidden_size
        # 从配置中获取注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否可以完全整除，否则抛出异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 计算缩放因子，用于缩放注意力分数
        self.scale = self.head_dim**-0.5
        # 从配置中获取注意力机制的dropout率
        self.dropout = config.attention_dropout

        # 初始化线性层，用于投影查询、键和值
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 输出投影层，用于最终的线性变换
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 将输入张量重塑为适当形状的私有方法
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接受隐藏状态和各种掩码作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Owlv2
class Owlv2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 存储配置对象
        self.activation_fn = ACT2FN[config.hidden_act]  # 设置激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 定义线性层 fc1
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 定义线性层 fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 应用第一个线性层
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 应用第二个线性层
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->Owlv2
class Owlv2EncoderLayer(nn.Module):
    def __init__(self, config: Owlv2Config):
        super().__init__()
        self.embed_dim = config.hidden_size  # 设置嵌入维度
        self.self_attn = Owlv2Attention(config)  # 创建自注意力层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层 1
        self.mlp = Owlv2MLP(config)  # 创建 MLP 层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层 2

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
        residual = hidden_states  # 保存残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 应用 LayerNorm 层 1
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )  # 应用自注意力机制
        hidden_states = residual + hidden_states  # 添加残差连接

        residual = hidden_states  # 更新残差连接

        hidden_states = self.layer_norm2(hidden_states)  # 应用 LayerNorm 层 2
        hidden_states = self.mlp(hidden_states)  # 应用 MLP 层
        hidden_states = residual + hidden_states  # 添加残差连接

        outputs = (hidden_states,)  # 准备输出结果

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs


# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTPreTrainedModel with OwlViT->Owlv2,owlvit->owlv2
class Owlv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 使用 Owlv2Config 类作为配置类
    config_class = Owlv2Config
    # 设置基础模型前缀为 "owlv2"
    base_model_prefix = "owlv2"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要进行模块拆分的模块列表
    _no_split_modules = ["Owlv2EncoderLayer"]

    # 初始化模型权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor

        # 如果模块是 Owlv2TextEmbeddings 类型
        if isinstance(module, Owlv2TextEmbeddings):
            # 初始化 token_embedding 和 position_embedding 的权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)

        # 如果模块是 Owlv2VisionEmbeddings 类型
        elif isinstance(module, Owlv2VisionEmbeddings):
            # 初始化 class_embedding, patch_embedding 和 position_embedding 的权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)

        # 如果模块是 Owlv2Attention 类型
        elif isinstance(module, Owlv2Attention):
            # 初始化 attention 模块的权重
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)

        # 如果模块是 Owlv2MLP 类型
        elif isinstance(module, Owlv2MLP):
            # 初始化 MLP 模块的权重
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        # 如果模块是 Owlv2Model 类型
        elif isinstance(module, Owlv2Model):
            # 初始化模型的 text_projection 和 visual_projection 的权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        # 如果模块是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 模块的偏置和权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 如果模块是 nn.Linear 类型且有偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将线性层的偏置初始化为零
            module.bias.data.zero_()
# OWLV2_START_DOCSTRING 是模型文档字符串的开始，描述了该模型从 PreTrainedModel 继承，提供了通用方法，
# 如下载或保存模型、调整输入嵌入大小、修剪头等。
OWLV2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Owvl2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# OWLV2_TEXT_INPUTS_DOCSTRING 是用于文本输入的文档字符串，描述了输入参数及其用途。
OWLV2_TEXT_INPUTS_DOCSTRING = r"""
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

# OWLV2_VISION_INPUTS_DOCSTRING 是用于视觉输入的文档字符串，描述了输入参数及其用途。
OWLV2_VISION_INPUTS_DOCSTRING = r"""
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

# OWLV2_INPUTS_DOCSTRING 是输入参数文档字符串的起始，作为一个整体，包含了不同输入类型的文档字符串。
OWLV2_INPUTS_DOCSTRING = r"""
"""  # 这是一个占位符字符串，用于组合不同输入类型的文档字符串。
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。可以使用 [`AutoTokenizer`] 获得这些索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 进行详细了解。[什么是输入 ID？](../glossary#input-ids)
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。
            Pixel values.
        
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 注意力掩码，用于避免在填充标记索引上执行注意力操作。掩码取值为 `[0, 1]`：
            # - 1 表示 **未被遮蔽** 的标记，
            # - 0 表示 **被遮蔽** 的标记。
            [什么是注意力掩码？](../glossary#attention-mask)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
            Whether or not to return the contrastive loss.
        
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。在返回的张量中查看 `attentions` 以获取更多详细信息。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。在返回的张量中查看 `hidden_states` 以获取更多详细信息。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        return_base_image_embeds (`bool`, *optional*):
            # 是否返回基础图像嵌入。
            Whether or not to return the base image embeddings.
        
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而非普通元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
# OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING 是一个文档字符串，描述了 OWLV2 模型输入的参数及其形状和含义
OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。
        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*):
            输入序列标记在词汇表中的索引。索引可以通过 [`AutoTokenizer`] 获取。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取更多详情。[什么是输入 ID？](../glossary#input-ids)。
        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
            遮罩，避免对填充的标记索引执行注意力。遮罩值选择在 `[0, 1]`：
            - 1 表示 **未被遮罩** 的标记，
            - 0 表示 **被遮罩** 的标记。
            [什么是注意力遮罩？](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            是否返回最后一个隐藏状态。查看返回张量中的 `text_model_last_hidden_state` 和 `vision_model_last_hidden_state` 获取更多详情。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING 是一个文档字符串，描述了 OWLV2 模型输入的参数及其形状和含义
OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。
        query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            要检测的查询图像的像素值。每个目标图像都传入一个查询图像。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 获取更多详情。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 获取更多详情。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 以下是一个自定义的编码器类 Owlv2Encoder，用于 OWLV2 模型
class Owlv2Encoder(nn.Module):
    """
    Transformer 编码器，包含 `config.num_hidden_layers` 个自注意力层。每一层是一个 [`Owlv2EncoderLayer`]。

    Args:
        config: Owlv2Config
    """

    def __init__(self, config: Owlv2Config):
        super().__init__()
        # 创建一个包含 `config.num_hidden_layers` 个 Owlv2EncoderLayer 的模块列表
        self.layers = nn.ModuleList([Owlv2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
        self.gradient_checkpointing = False
    # 定义一个方法 `forward`，用于执行模型的前向传播
    def forward(
        self,
        # 输入的嵌入向量，可以是任意形状的张量
        inputs_embeds,
        # 注意力掩码，用于指示哪些位置需要注意，可以是可选的张量
        attention_mask: Optional[torch.Tensor] = None,
        # 因果注意力掩码，用于自回归任务中，指示哪些位置不应被关注
        causal_attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回一个字典作为输出
        return_dict: Optional[bool] = None,
# 从 transformers.models.owlvit.modeling_owlvit.OwlViTTextTransformer 复制而来，将 OWLVIT 替换为 OWLV2，OwlViT 替换为 Owlv2
class Owlv2TextTransformer(nn.Module):
    # 初始化函数，接收 Owlv2TextConfig 类型的参数 config
    def __init__(self, config: Owlv2TextConfig):
        super().__init__()
        self.config = config
        # 根据配置的 hidden_size 获取嵌入维度
        embed_dim = config.hidden_size
        # 初始化 Owlv2TextEmbeddings 对象，用于处理输入的嵌入
        self.embeddings = Owlv2TextEmbeddings(config)
        # 初始化 Owlv2Encoder 对象，用于编码输入序列
        self.encoder = Owlv2Encoder(config)
        # 初始化 LayerNorm 层，用于最终的归一化处理
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 定义前向传播函数，添加 OWLV2_TEXT_INPUTS_DOCSTRING 的模型输入文档注释
    # 并使用 replace_return_docstrings 将输出类型替换为 BaseModelOutputWithPooling，配置类为 Owlv2TextConfig
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
        """
        # 如果未显式提供，则使用配置中的输出注意力机制设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未显式提供，则使用配置中的输出隐藏状态设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未显式提供，则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入的张量形状
        input_shape = input_ids.size()
        # 将输入的张量重新视图为二维张量，其中的最后一个维度保持不变
        input_ids = input_ids.view(-1, input_shape[-1])
        # 使用嵌入层将输入张量和位置编码作为参数传入，生成隐藏状态
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # num_samples, seq_len = input_shape  where num_samples = batch_size * num_max_text_queries
        # OWLV2 的文本模型使用因果注意力掩码，在此处准备它
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        # 创建一个四维因果注意力掩码，基于输入形状、隐藏状态的数据类型和设备
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # 如果存在注意力掩码，则扩展它的维度
        if attention_mask is not None:
            # 将二维注意力掩码扩展为四维格式 [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 将隐藏状态作为输入嵌入传递给编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出中的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后一个隐藏状态进行最终的层归一化
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 从tokens嵌入的末尾获取特征（每个序列中最大的数值对应的位置）
        # 为了兼容ONNX，将input_ids转换为torch.int类型进行argmax操作
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        # 如果不使用返回字典模式，则返回多个输出项的元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用BaseModelOutputWithPooling类封装结果，以返回更结构化的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从 transformers.models.owlvit.modeling_owlvit.OwlViTTextModel 复制到 Owlv2TextModel，并更新了一些路径和类名
class Owlv2TextModel(Owlv2PreTrainedModel):
    # 指定配置类为 Owlv2TextConfig
    config_class = Owlv2TextConfig

    def __init__(self, config: Owlv2TextConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 使用 Owlv2TextTransformer 创建文本模型
        self.text_model = Owlv2TextTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的输入嵌入层（token_embedding）
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        # 设置文本模型的输入嵌入层（token_embedding）
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2TextConfig)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        前向传播方法
        
        Returns:
            返回一个元组或者 BaseModelOutputWithPooling 类型的对象
        """
        # 调用文本模型的前向传播方法，传入输入参数
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 从 transformers.models.owlvit.modeling_owlvit.OwlViTVisionTransformer 复制到 Owlv2VisionTransformer，并更新了一些路径和类名
class Owlv2VisionTransformer(nn.Module):
    def __init__(self, config: Owlv2VisionConfig):
        super().__init__()
        self.config = config

        # 初始化视觉嵌入层和前层归一化层
        self.embeddings = Owlv2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化编码器和后层归一化层
        self.encoder = Owlv2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        神经网络模型的前向传播函数，接收输入像素值和一些可选参数，并返回模型输出。

        Args:
            pixel_values (torch.FloatTensor): 输入的像素值张量。
            output_attentions (Optional[bool]): 是否输出注意力权重，默认为None。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态，默认为None。
            return_dict (Optional[bool]): 是否以字典形式返回输出，默认为None。

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 根据 `return_dict` 参数返回不同形式的模型输出。

        """
        # 根据传入的参数或配置选择是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入的参数或配置选择是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入的参数或配置选择是否使用返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入像素值转换为预期的数据类型
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        # 嵌入层处理输入像素值
        hidden_states = self.embeddings(pixel_values)
        # 应用预层归一化
        hidden_states = self.pre_layernorm(hidden_states)

        # 编码器处理嵌入后的隐藏状态
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态和池化输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        # 应用后层归一化到池化输出
        pooled_output = self.post_layernorm(pooled_output)

        # 根据 `return_dict` 参数返回不同形式的模型输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用自定义的输出对象构建返回结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从 transformers.models.owlvit.modeling_owlvit.OwlViTVisionModel 复制并修改为 Owlv2VisionModel，包括 OWLVIT->OWLV2,OwlViT->Owlv2,google/owlvit-base-patch32->google/owlv2-base-patch16 的变更
class Owlv2VisionModel(Owlv2PreTrainedModel):
    # 设置配置类为 Owlv2VisionConfig
    config_class = Owlv2VisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: Owlv2VisionConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 初始化视觉模型为 Owlv2VisionTransformer
        self.vision_model = Owlv2VisionTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型的嵌入层中的 patch_embedding
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回视觉模型的前向传播结果。

        参数:
        - pixel_values: 可选的 torch.FloatTensor，像素值
        - output_attentions: 可选的 bool，是否输出注意力权重
        - output_hidden_states: 可选的 bool，是否输出隐藏状态
        - return_dict: 可选的 bool，是否返回字典形式的输出

        返回:
        - BaseModelOutputWithPooling 或 Tuple，模型输出的汇总结果

        示例:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Owlv2VisionModel

        >>> model = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # 池化后的 CLS 状态
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(OWLV2_START_DOCSTRING)
# 从 transformers.models.owlvit.modeling_owlvit.OwlViTModel 复制并修改为 Owlv2Model，包括 google/owlvit-base-patch32->google/owlv2-base-patch16-ensemble, OWLVIT->OWLV2,OwlViT->Owlv2,owlvit->owlv2,OWL-ViT->OWLv2 的变更
class Owlv2Model(Owlv2PreTrainedModel):
    # 设置配置类为 Owlv2Config
    config_class = Owlv2Config
    def __init__(self, config: Owlv2Config):
        # 调用父类构造函数初始化
        super().__init__(config)

        # 检查配置文件中的文本配置是否为Owlv2TextConfig类型，若不是则抛出数值错误异常
        if not isinstance(config.text_config, Owlv2TextConfig):
            raise ValueError(
                "config.text_config is expected to be of type Owlv2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查配置文件中的视觉配置是否为Owlv2VisionConfig类型，若不是则抛出数值错误异常
        if not isinstance(config.vision_config, Owlv2VisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type Owlv2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从配置文件中获取文本和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度，文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = Owlv2TextTransformer(text_config)
        self.vision_model = Owlv2VisionTransformer(vision_config)

        # 设置视觉投影和文本投影层，不带偏置
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 设置logit缩放作为模型参数
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`Owlv2TextModel`].

        Examples:
        ```python
        >>> from transformers import AutoProcessor, Owlv2Model

        >>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astronaut"]], return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 如果未指定返回字典，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取所有批次样本中所有文本查询的嵌入
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        # 提取池化输出作为文本特征
        pooled_output = text_output[1]
        # 将池化输出投影到文本投影层
        text_features = self.text_projection(pooled_output)

        return text_features
    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`Owlv2VisionModel`].

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Owlv2Model

        >>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        
        # Use OWLv2 model's config for some fields (if specified) instead of those of vision & text components.
        # 根据需要，使用 OWLv2 模型的配置替换视觉和文本组件的相关字段。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 调用视觉模型来获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉模型的输出中获取池化后的输出
        pooled_output = vision_outputs[1]
        # 使用视觉投影层对池化输出进行投影，得到图像特征
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征
        return image_features

    @add_start_docstrings_to_model_forward(OWLV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2Output, config_class=Owlv2Config)
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
        # 与 get_image_features 方法类似，该方法在模型的正向传播过程中处理输入和输出
        # 具体实现的细节会因模型的不同而有所不同，这里给出了一个大致的框架和输入参数
# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTBoxPredictionHead with OwlViT->Owlv2
class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config, out_dim: int = 4):
        super().__init__()

        # Extract hidden size from configuration
        width = config.vision_config.hidden_size
        # Define fully connected layers for box prediction head
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        # Pass through the first linear layer followed by GELU activation
        output = self.dense0(image_features)
        output = self.gelu(output)
        # Pass through the second linear layer followed by GELU activation
        output = self.dense1(output)
        output = self.gelu(output)
        # Final prediction through the third linear layer
        output = self.dense2(output)
        return output


# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTClassPredictionHead with OwlViT->Owlv2
class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config):
        super().__init__()

        # Extract hidden sizes from configuration
        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size

        # Define fully connected layers and activation functions for class prediction head
        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        # Compute image class embeddings
        image_class_embeds = self.dense0(image_embeds)

        # Handle case when query embeddings are not provided
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            # Initialize prediction logits with zeros
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and query embeddings
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Calculate class predictions using matrix multiplication
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        # Apply mask to logits if provided
        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)


class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    config_class = Owlv2Config
    # 初始化函数，接收一个 Owlv2Config 类型的参数 config
    def __init__(self, config: Owlv2Config):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)

        # 创建 Owlv2Model 模型对象，使用传入的 config 参数
        self.owlv2 = Owlv2Model(config)
        # 创建 Owlv2ClassPredictionHead 类对象，使用传入的 config 参数
        self.class_head = Owlv2ClassPredictionHead(config)
        # 创建 Owlv2BoxPredictionHead 类对象，使用传入的 config 参数
        self.box_head = Owlv2BoxPredictionHead(config)
        # 创建 Owlv2BoxPredictionHead 类对象，使用传入的 config 参数，设置 out_dim=1
        self.objectness_head = Owlv2BoxPredictionHead(config, out_dim=1)

        # 创建一个 LayerNorm 层，使用 config 中的 hidden_size 和 layer_norm_eps 参数
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        # 创建一个 Sigmoid 激活函数对象
        self.sigmoid = nn.Sigmoid()

        # 计算 sqrt_num_patches，即图像尺寸除以补丁尺寸，结果取整
        self.sqrt_num_patches = config.vision_config.image_size // config.vision_config.patch_size

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.normalize_grid_corner_coordinates 复制而来
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # 计算特征图中每个 patch 的归一化的 xy 角落坐标

        # 检查 feature_map 的维度是否为 4
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        # 获取 feature_map 的设备信息
        device = feature_map.device
        num_patches = feature_map.shape[1]

        # 使用 numpy 创建二维坐标网格，表示每个 patch 的角落坐标，结果类型为 float32
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # 将 (h, w, 2) 的坐标数组展平为 (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        # 将 numpy 数组转换为 torch 张量，并发送到与 feature_map 相同的设备
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    # 预测每个图像特征 token 是否是对象的概率
    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        # 对输入的 image_features 进行去除梯度操作
        image_features = image_features.detach()
        # 使用 objectness_head 对 image_features 进行预测，得到对象性得分
        objectness_logits = self.objectness_head(image_features)
        # 从 objectness_logits 中提取第一个维度的数据，通常表示对象性的得分
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.compute_box_bias 复制而来
    # 计算盒子中心相对于特征网格位置的偏置
    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # 将盒子坐标规范化到网格角落坐标
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        # 将坐标裁剪到区间 [0.0, 1.0]
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # 反归一化 xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # 计算盒子尺寸相对于补丁尺寸的偏置
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # 计算盒子偏置
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.box_predictor 复制而来
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                从 image_text_embedder 方法返回的图像提取特征。
            feature_map:
                图像特征的空间重新排列，也是从 image_text_embedder 方法返回的。

        Returns:
            pred_boxes:
                预测框列表 (cxcywh，归一化到 0 到 1)，嵌套在一个字典中。
        """
        # 边界框检测头 [batch_size, num_boxes, 4]。
        pred_boxes = self.box_head(image_feats)

        # 计算每个令牌在网格上的位置，并用它来计算bbox预测的偏置
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.class_predictor 复制而来
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                从 image_text_embedder 提取的特征。
            query_embeds:
                文本查询嵌入。
            query_mask:
                必须与 query_embeddings 一起提供。指示哪些查询嵌入是有效的掩码。

        Returns:
            (pred_logits, image_class_embeds):
                预测的逻辑张量和图像类别嵌入。
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.image_text_embedder 复制而来，owlvit 改为 owlv2
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            input_ids:
                输入的令牌 IDs。
            pixel_values:
                图像的像素值。
            attention_mask:
                用于指示输入的注意力掩码。
            output_attentions:
                是否输出注意力权重。
            output_hidden_states:
                是否输出隐藏状态。

        Returns:
            (sequence_output, pooled_output):
                序列输出和池化输出。
        """
        # 实现在模型中嵌入图像和文本的函数
        raise NotImplementedError
    ) -> Tuple[torch.FloatTensor]:
        # 对文本和图像进行编码

        # 使用 OwlV2 模型进行处理，传入像素值、输入 ID、注意力掩码等参数
        outputs = self.owlv2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # 获取图像嵌入
        # 从 OwlV2 模型的输出中提取最后隐藏状态
        last_hidden_state = outputs.vision_model_output[0]
        # 应用后层归一化到图像嵌入
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)

        # 调整类别令牌的大小
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # 将图像嵌入与类别令牌合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        # 应用层归一化到图像嵌入
        image_embeds = self.layer_norm(image_embeds)

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        # 从 OwlV2 模型的输出中提取文本嵌入
        text_embeds = outputs[-4]

        return (text_embeds, image_embeds, outputs)

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.image_embedder 复制而来，将 owlvit->owlv2，OwlViTModel->Owlv2Model
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # 获取 Owlv2Model 视觉嵌入（与 CLIP 相同）

        # 使用 OwlV2 模型处理像素值，返回字典
        vision_outputs = self.owlv2.vision_model(pixel_values=pixel_values, return_dict=True)

        # 应用后层归一化到最后隐藏状态，返回非投影输出
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)

        # 调整类别令牌的大小
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # 将图像嵌入与类别令牌合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        # 应用层归一化到图像嵌入
        image_embeds = self.layer_norm(image_embeds)

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.embed_image_query 复制而来
    def embed_image_query(
        self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 获取类别预测的结果，但不使用
        _, class_embeds = self.class_predictor(query_image_features)
        # 使用查询图像特征进行边界框预测
        pred_boxes = self.box_predictor(query_image_features, query_feature_map)
        # 将预测的边界框转换为左上角和右下角坐标格式
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)

        # 遍历查询图像
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device

        for i in range(query_image_features.shape[0]):
            # 创建一个形状为 [1, 4] 的张量，用于表示每个查询框
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            # 获取当前查询图像的预测边界框
            each_query_pred_boxes = pred_boxes_as_corners[i]
            # 计算每个查询框与预测边界框之间的 IoU
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # 如果没有重叠的框，则使用广义 IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # 使用自适应阈值选取IoU最佳80%范围内的所有框
            iou_threshold = torch.max(ious) * 0.8

            # 找到满足阈值条件的预测框索引
            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                # 选取类别嵌入向量
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                # 计算选取类别嵌入向量的平均值
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                # 计算平均相似度
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                # 选择平均相似度最小的预测框索引
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        # 如果存在最佳类别嵌入向量，则堆叠它们
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        # 返回查询嵌入向量、预测框索引和预测框
        return query_embeds, box_indices, pred_boxes

    @add_start_docstrings_to_model_forward(OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2Config)
    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 实现 OWLV2 模型的图像引导目标检测

    @add_start_docstrings_to_model_forward(OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ObjectDetectionOutput, config_class=Owlv2Config)
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # OWLV2 模型的前向传播
```