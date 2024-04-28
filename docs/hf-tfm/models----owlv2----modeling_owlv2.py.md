# `.\transformers\models\owlv2\modeling_owlv2.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 基于 Apache License Version 2.0 授权
# 只有在符合许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律有要求或书面同意，否则不进行软件分发
# 软件按"原样"提供，没有任何明示或暗示的担保或条件
# 有关特定语言的权限和限制条件，请参阅许可证
"PyTorch OWLv2 model."

# 导入模块
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
# 导入自定义模块
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
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig

# 如果视觉模块可用，则导入图像转换模块
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点，指向 OWLv2 模型
_CHECKPOINT_FOR_DOC = "google/owlv2-base-patch16-ensemble"

# 查看所有 OWLv2 模型的存档列表
OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlv2-base-patch16-ensemble",
    # 可在 https://huggingface.co/models?filter=owlv2 查看所有 OWLv2 模型
]

# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制，将 clip->owlv2
# 定义对比损失函数
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 从 transformers.models.clip.modeling_clip.clip_loss 复制，将 clip->owlv2
# 定义 OWLv2 损失函数
def owlv2_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算标题损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像损失
    image_loss = contrastive_loss(similarity.t())
    # 返回标题损失和图像损失的平均值
    return (caption_loss + image_loss) / 2.0

# 定义 OWLv2 输出数据类
@dataclass
class Owlv2Output(ModelOutput):
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

    # 定义一些可选的属性，初始化为 None
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 将类的属性转换为元组形式，用于进一步处理
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果属性不是 "text_model_output" 或 "vision_model_output"，则直接返回属性值
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()  # 遍历类的所有属性键
        )
# 从 transformers.models.detr.modeling_detr._upcast 复制的函数
def _upcast(t: Tensor) -> Tensor:
    # 通过将数据类型提升到等效更高的类型来避免在乘法运算中发生数值溢出
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 从 transformers.models.detr.modeling_detr.box_area 复制的函数
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应为 (x1, y1, x2, y2) 格式，其中 `0 <= x1 < x2` 且 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框的面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 从 transformers.models.detr.modeling_detr.box_iou 复制的函数
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


# 从 transformers.models.detr.modeling_detr.generalized_box_iou 复制的函数
def generalized_box_iou(boxes1, boxes2):
    """
    来自 https://giou.stanford.edu/ 的广义 IoU。边界框应为 [x0, y0, x1, y1]（角）格式。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1) 且 M = len(boxes2)
    """
    # 对于退化的边界框，会导致 inf / nan 结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须为 [x0, y0, x1, y1]（角）格式，但得到了 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须为 [x0, y0, x1, y1]（角）格式，但得到了 {boxes2}")
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

    # 初始化变量，用于存储 loss 相关数据
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    objectness_logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    # 将当前对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        # 使用列表推导式遍历对象的键
        return tuple(
            # 如果键不是 "text_model_output" 或 "vision_model_output"，则直接返回对应的值
            self[k] if k not in ["text_model_output", "vision_model_output"] else
            # 否则，调用对应属性的 to_tuple() 方法获取元组
            getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 这个数据类定义了 Owlv2ImageGuidedObjectDetectionOutput 类，它是 ModelOutput 的子类
# 这个类包含了 Owlv2 for Object Detection 模型的输出结果
@dataclass
class Owlv2ImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    # 分类logits，包括无目标类的logits
    logits: torch.FloatTensor = None
    # 目标图像的归一化边框坐标
    target_pred_boxes: torch.FloatTensor = None
    # 查询图像的归一化边框坐标
    query_pred_boxes: torch.FloatTensor = None
    # 图像patch的嵌入
    image_embeds: torch.FloatTensor = None
    # 查询图像patch的嵌入
    query_image_embeds: torch.FloatTensor = None
    # 图像patch的类嵌入
    class_embeds: torch.FloatTensor = None
    # 文本模型的输出
    text_model_output: BaseModelOutputWithPooling = None
    # 视觉模型的输出
    vision_model_output: BaseModelOutputWithPooling = None
    # 将对象转换为元组的方法，返回一个元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 对于对象自身的键值对，如果键不是"text_model_output"或"vision_model_output"，
            # 则直接取值；否则调用getattr方法获取对应属性的值，并调用属性的to_tuple方法
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            # 遍历对象自身的所有键
            for k in self.keys()
        )
# 从transformers.models.owlvit.modeling_owlvit.OwlViTVisionEmbeddings复制代码，并将OwlViT->Owlv2
class Owlv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Owlv2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# 从transformers.models.owlvit.modeling_owlvit.OwlViTTextEmbeddings复制代码，并将OwlViT->Owlv2
class Owlv2TextEmbeddings(nn.Module):
    def __init__(self, config: Owlv2TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
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

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# 从transformers.models.owlvit.modeling_owlvit.OwlViTAttention复制代码，并将OwlViT->Owlv2
class Owlv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    # 初始化函数，接受配置参数，并调用父类初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置参数
        self.config = config
        # 设置嵌入维度为配置参数中的隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头数为配置参数中的注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查嵌入维度是否能被注意力头数整除
        if self.head_dim * self.num_heads != self.embed_dim:
            # 如果不能整除，抛出数值错误异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子为头维度的倒数的平方
        self.scale = self.head_dim**-0.5
        # 设置注意力的丢弃率
        self.dropout = config.attention_dropout

        # 初始化线性变换层用于键、值、查询和输出投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 调整张量形状以便进行多头注意力计算
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量形状重塑为 (bsz, seq_len, num_heads, head_dim) 并交换维度顺序
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# 定义 Owlv2MLP 类，它是一个由全连接层和激活函数组成的多层感知机
class Owlv2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义两个全连接层
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过第一个全连接层和激活函数
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        # 经过第二个全连接层
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 定义 Owlv2EncoderLayer 类，它包含自注意力机制和 MLP 层
class Owlv2EncoderLayer(nn.Module):
    def __init__(self, config: Owlv2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        # 定义自注意力机制
        self.self_attn = Owlv2Attention(config)
        # 定义第一个层归一化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 定义 MLP 层
        self.mlp = Owlv2MLP(config)
        # 定义第二个层归一化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        # 保存残差连接
        residual = hidden_states

        # 经过第一个层归一化层和自注意力机制
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 经过第二个层归一化层和 MLP 层
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 返回输出
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# 定义 Owlv2PreTrainedModel 类，它是一个抽象类，用于处理权重初始化和加载预训练模型
class Owlv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 定义 Owlv2Config 作为配置类
    config_class = Owlv2Config
    # 定义 "owlv2" 作为基础模型前缀
    base_model_prefix = "owlv2"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块
    _no_split_modules = ["Owlv2EncoderLayer"]
    
    # 初始化权重的函数
    def _init_weights(self, module):
        """初始化权重"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果是 Owlv2TextEmbeddings 模块
        if isinstance(module, Owlv2TextEmbeddings):
            # 初始化token和位置embedding
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果是 Owlv2VisionEmbeddings 模块
        elif isinstance(module, Owlv2VisionEmbeddings):
            # 初始化class embedding、patch embedding和位置embedding
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果是 Owlv2Attention 模块
        elif isinstance(module, Owlv2Attention):
            # 初始化q/k/v projection和out projection
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果是 Owlv2MLP 模块
        elif isinstance(module, Owlv2MLP):
            # 初始化全连接层
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果是 Owlv2Model 模块
        elif isinstance(module, Owlv2Model):
            # 初始化text和visual projection
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        # 如果是 LayerNorm 模块
        if isinstance(module, nn.LayerNorm):
            # 初始化LayerNorm的偏置和权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 Linear 模块且有偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 初始化Linear的偏置为0
            module.bias.data.zero_()
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
# OWLV2_START_DOCSTRING: 模型的开始文档字符串，包含了模型的继承关系、用法说明以及参数说明

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
# OWLV2_TEXT_INPUTS_DOCSTRING: 文本输入参数文档字符串，包含了输入参数的说明，如input_ids、attention_mask等

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
# OWLV2_VISION_INPUTS_DOCSTRING: 视觉输入参数文档字符串，包含了输入参数的说明，如pixel_values、output_attentions等

OWLV2_INPUTS_DOCSTRING = r"""
"""
# OWLV2_INPUTS_DOCSTRING: 输入参数文档字符串，用于组合文本输入和视觉输入参数的说明
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。可以使用 [`AutoTokenizer`] 获得索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。[什么是输入 ID？](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充的标记索引上执行注意力的掩码。掩码值选取在 `[0, 1]` 范围内:
            # - 对于**未屏蔽**的标记，值为 1，
            # - 对于**屏蔽**的标记，值为 0。
            [什么是注意力掩码？](../glossary#attention-mask)
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请查看返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请查看返回张量下的 `hidden_states`。
        return_base_image_embeds (`bool`, *optional*):
            # 是否返回基础图像嵌入。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是简单的元组。
# 定义用于目标检测的输入数据的文档字符串
OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            输入图像的像素值。
        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*):
            输入序列的词汇表索引。可以使用 `AutoTokenizer` 获得。
        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
            注意力掩码。用于避免在填充标记上执行注意力。
        output_hidden_states (`bool`, *optional*):
            是否返回最后一层的隐藏状态。
        return_dict (`bool`, *optional*):
            是否返回 `~utils.ModelOutput` 而不是普通元组。
"""

# 定义用于图像引导目标检测的输入数据的文档字符串
OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            输入图像的像素值。
        query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            查询图像的像素值。每个目标图像传入一个查询图像。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。
        return_dict (`bool`, *optional*):
            是否返回 `~utils.ModelOutput` 而不是普通元组。
"""

# 定义 Owlv2Encoder 类
# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTEncoder with OwlViT->Owlv2
class Owlv2Encoder(nn.Module):
    """
    Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每个层都是 [`Owlv2EncoderLayer`]。

    Args:
        config: Owlv2Config
    """

    def __init__(self, config: Owlv2Config):
        super().__init__()
        # 创建 Owlv2EncoderLayer 的列表
        self.layers = nn.ModuleList([Owlv2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False
    # 前向传播方法
    def forward(
        # 输入的embeddings
        self,
        inputs_embeds,
        # 注意力掩码，指定哪些位置参与注意力计算
        attention_mask: Optional[torch.Tensor] = None,
        # 因果注意力掩码，用于自回归模型
        causal_attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出中间隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回PyTorch原生的SequenceOutput对象
        return_dict: Optional[bool] = None,
# 从transformers.models.owlvit.modeling_owlvit.OwlViTTextTransformer复制而来，将OWLVIT->OWLV2，OwlViT->Owlv2
class Owlv2TextTransformer(nn.Module):
    def __init__(self, config: Owlv2TextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Owlv2TextEmbeddings(config)  # 创建文本嵌入对象
        self.encoder = Owlv2Encoder(config)  # 创建编码器对象
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 创建最终的层归一化对象

    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)  # 添加模型前向传播的起始文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2TextConfig)  # 替换返回的文档字符串
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ...
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入张量的形状
        input_shape = input_ids.size()
        # 将输入张量的维度转换为二维
        input_ids = input_ids.view(-1, input_shape[-1])
        # 使用输入张量和位置编码获取隐藏状态
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 创建因果注意力掩码
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # 如果提供了注意力掩码，需要进行扩展
        if attention_mask is not None:
            # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态并进行标准化处理
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 从令牌嵌入的末尾获取特征
        # 为了在onnx中兼容，需要将argmax的输出转换为torch.int，因为opset 14不支持int64输入
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        # 如果不返回字典形式的结果，则返回元组形式
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回以 BaseModelOutputWithPooling 类型封装的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 基于transformers.models.owlvit.modeling_owlvit.OwlViTTextModel，进行了一些修改，将OWLVIT改为OWLV2，OwlViT改为Owlv2
class Owlv2TextModel(Owlv2PreTrainedModel):
    config_class = Owlv2TextConfig

    def __init__(self, config: Owlv2TextConfig):
        super().__init__(config)
        # 初始化文本模型
        self.text_model = Owlv2TextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # OwlV2文本模型的前向传播方法
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
        r"""
        返回：

        示例:
        ```python
        >>> from transformers import AutoProcessor, Owlv2TextModel

        >>> model = Owlv2TextModel.from_pretrained("google/owlv2-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # 获取所有批次样本中所有文本查询的嵌入
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 从transformers.models.owlvit.modeling_owlvit.OwlViTVisionTransformer中复制并修改了一些内容，将OWLVIT改为OWLV2，OwlViT改为Owlv2
class Owlv2VisionTransformer(nn.Module):
    def __init__(self, config: Owlv2VisionConfig):
        super().__init__()
        self.config = config

        # 初始化视觉嵌入
        self.embeddings = Owlv2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Owlv2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # OwlV2视觉转换器的前向传播方法
    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfig)
    # 定义一个名为forward的方法，用于模型的前向传播
    def forward(
        self,
        # 输入像素值的张量，数据类型为torch.FloatTensor
        pixel_values: torch.FloatTensor,
        # 是否输出注意力权重，默认为空
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为空
        output_hidden_states: Optional[bool] = None,
        # 是否返回结果字典，默认为空
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        # 如果output_attentions不为空，则使用给定值，否则使用self.config.output_attentions的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states不为空，则使用给定值，否则使用self.config.output_hidden_states的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict不为空，则使用给定值，否则使用self.config.use_return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据类型转换为期望的dtype
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        # 通过嵌入层处理输入像素值
        hidden_states = self.embeddings(pixel_values)
        # 对隐层状态进行layer norm
        hidden_states = self.pre_layernorm(hidden_states)

        # 使用编码器进行推理
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏层状态和池化输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        # 对池化输出进行layer norm
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不需要返回结果字典，则返回相应的元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回包含隐藏状态、池化输出、隐藏状态和注意力权重的结果字典对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从transformers.models.owlvit.modeling_owlvit.OwlViTVisionModel复制而来，将OWLVIT->OWLV2，OwlViT->Owlv2，google/owlvit-base-patch32->google/owlv2-base-patch16
# 定义Owlv2VisionModel类，继承自Owlv2PreTrainedModel类
class Owlv2VisionModel(Owlv2PreTrainedModel):
    # 定义config_class为Owlv2VisionConfig
    config_class = Owlv2VisionConfig
    # 定义主输入名称为"pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受config参数为Owlv2VisionConfig类型
    def __init__(self, config: Owlv2VisionConfig):
        super().__init__(config)
        # 初始化vision_model为Owlv2VisionTransformer对象
        self.vision_model = Owlv2VisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法，接受pixel_values, output_attentions, output_hidden_states, return_dict等参数
    # 返回Union[Tuple, BaseModelOutputWithPooling]类型值
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
        返回值：

        示例：
        ```
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
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 调用Owlv2VisionTransformer的前向传播方法
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 添加注释到OWLV2_START_DOCSTRING
# 从transformers.models.owlvit.modeling_owlvit.OwlViTModel复制而来，将google/owlvit-base-patch32->google/owlv2-base-patch16-ensemble，OWLVIT->OWLV2，OwlViT->Owlv2，owlvit->owlv2，OWL-ViT->OWLv2
# 定义Owlv2Model类，继承自Owlv2PreTrainedModel类
class Owlv2Model(Owlv2PreTrainedModel):
    # 定义config_class为Owlv2Config
    config_class = Owlv2Config
    # 初始化函数，接受一个 Owlv2Config 对象作为参数
    def __init__(self, config: Owlv2Config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查 config.text_config 是否为 Owlv2TextConfig 类型，不是则抛出数值错误
        if not isinstance(config.text_config, Owlv2TextConfig):
            raise ValueError(
                "config.text_config is expected to be of type Owlv2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 Owlv2VisionConfig 类型，不是则抛出数值错误
        if not isinstance(config.vision_config, Owlv2VisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type Owlv2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取 text_config 和 vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = Owlv2TextTransformer(text_config)
        self.vision_model = Owlv2VisionTransformer(vision_config)

        # 初始化视觉投影和文本投影
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加文档字符串到模型前向传播函数
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
        返回文本特征，通过将投影层应用到 Owlv2TextModel 的池化输出而获得。

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
        # 如果未指定 return_dict，则使用模型的配置来决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取所有文本查询在所有批次样本中的嵌入
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        pooled_output = text_output[1]
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
        # 如果指定了output_attentions，则使用指定的值，否则使用OWLv2模型的配置值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了output_hidden_states，则使用指定的值，否则使用OWLv2模型的配置值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了return_dict，则使用指定的值，否则使用OWLv2模型的配置值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用vision_model方法，传入参数，并获取输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从vision_outputs中获取池化输出
        pooled_output = vision_outputs[1]
        # 通过visual_projection将池化输出投影得到图像特征
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
# 从 transformers.models.owlvit.modeling_owlvit.OwlViTBoxPredictionHead 复制并修改为 Owlv2BoxPredictionHead
class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config, out_dim: int = 4):
        super().__init__()

        # 获取视觉配置中的隐藏尺寸
        width = config.vision_config.hidden_size
        # 创建线性层，输入和输出维度都是隐藏尺寸
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        # 创建 GELU 激活函数层
        self.gelu = nn.GELU()
        # 创建线性层，输入是隐藏尺寸，输出是目标尺寸
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        # 通过第一个线性层传递图像特征
        output = self.dense0(image_features)
        # 应用 GELU 激活函数
        output = self.gelu(output)
        # 通过第二个线性层传递输出
        output = self.dense1(output)
        # 应用 GELU 激活函数
        output = self.gelu(output)
        # 通过第三个线性层传递输出
        output = self.dense2(output)
        return output


# 从 transformers.models.owlvit.modeling_owlvit.OwlViTClassPredictionHead 复制并修改为 Owlv2ClassPredictionHead
class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config):
        super().__init__()

        # 获取文本配置中的隐藏尺寸作为输出维度
        out_dim = config.text_config.hidden_size
        # 获取视觉配置中的隐藏尺寸作为查询维度
        self.query_dim = config.vision_config.hidden_size

        # 创建线性层，输入是查询维度，输出是文本隐藏尺寸
        self.dense0 = nn.Linear(self.query_dim, out_dim)
        # 创建线性层，输入是查询维度，输出是1
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        # 创建 ELU 激活函数层
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        # 通过第一个线性层传递图像嵌入
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            # 创建全零张量作为预测对数和图像类别嵌入
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # 归一化图像和文本特征
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # 获取类别预测
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # 应用可学习的移位和缩放到预测对数
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            # 将预测对数中的未知掩码位置设为负无穷大
            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)


# 为对象检测定义 Owlv2 类
class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    # 配置类为 Owlv2Config
    config_class = Owlv2Config
    # 初始化函数，接受一个Owlv2Config类型的config参数
    def __init__(self, config: Owlv2Config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建Owlv2Model对象
        self.owlv2 = Owlv2Model(config)
        # 创建Owlv2ClassPredictionHead对象
        self.class_head = Owlv2ClassPredictionHead(config)
        # 创建Owlv2BoxPredictionHead对象
        self.box_head = Owlv2BoxPredictionHead(config)
        # 创建一个只输出1维的Owlv2BoxPredictionHead对象
        self.objectness_head = Owlv2BoxPredictionHead(config, out_dim=1)

        # 创建一个LayerNorm对象
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        # 创建一个Sigmoid对象
        self.sigmoid = nn.Sigmoid()

    # 从transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection中复制的函数，对feature_map进行归一化并返回box的坐标
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # 根据feature_map计算归一化的xy坐标
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        # 获取feature_map所在的设备
        device = feature_map.device
        num_patches = feature_map.shape[1]

        # 计算box的坐标并归一化
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # 将(h, w, 2)形状的数组展平成(h*w, 2)形状
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        # 返回box的坐标
        return box_coordinates

    # 预测每个图像特征token是否是一个对象，返回对象性别分数
    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        # 从计算图中分离出image_features
        image_features = image_features.detach()
        # 使用objectness_head对image_features进行预测
        objectness_logits = self.objectness_head(image_features)
        # 提取出logits的第一个维度
        objectness_logits = objectness_logits[..., 0]
        # 返回对象性别分数
        return objectness_logits

    # 从transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection中复制的函数，计算box的偏差
    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # 对box的中心偏移到其在特征网格上的位置
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        # 将box坐标限制在0到1之间
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # 对xy进行反归一化
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # 将box大小偏移至patch的大小
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # 计算box偏差
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias
    # 从 OwlViTForObjectDetection.box_predictor 复制而来，定义了检测框的预测器
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                图像提取的特征，由 `image_text_embedder` 方法返回。
            feature_map:
                图像特征的空间重排列，也由 `image_text_embedder` 方法返回。
        Returns:
            pred_boxes:
                预测框的列表（归一化为 0 到 1 的 cxcywh 格式），嵌套在一个字典中。
        """
        # 检测框头部 [batch_size, num_boxes, 4]。
        pred_boxes = self.box_head(image_feats)

        # 计算每个标记在网格上的位置，并用它来计算 bbox 预测的偏置
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    # 从 OwlViTForObjectDetection.class_predictor 复制而来，定义了类别预测器
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                从 `image_text_embedder` 提取的特征。
            query_embeds:
                文本查询的嵌入。
            query_mask:
                必须与 query_embeddings 一起提供。指示哪些查询嵌入是有效的掩码。
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    # 从 OwlViTForObjectDetection.image_text_embedder 复制而来，定义了图像文本嵌入器，将 owlvit 改为 owlv2
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    # 定义函数，返回值为包含文本和图像编码的元组
    ) -> Tuple[torch.FloatTensor]:
        # 编码文本和图像
        outputs = self.owlv2(
            pixel_values=pixel_values,  # 像素值
            input_ids=input_ids,  # 输入标识符
            attention_mask=attention_mask,  # 注意力掩码
            output_attentions=output_attentions,  # 输出注意力
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=True,  # 返回字典格式结果
        )

        # 获取图像嵌入
        last_hidden_state = outputs.vision_model_output[0]  # 最后一个隐藏状态
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)  # 图像嵌入后的层归一化

        # 调整类别令牌的大小
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))  # 新尺寸
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)  # 广播类别令牌

        # 将图像嵌入与类别令牌合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out  # 图像嵌入与类别令牌相乘
        image_embeds = self.layer_norm(image_embeds)  # 图像嵌入层归一化

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)  # 重塑图像嵌入
        text_embeds = outputs[-4]  # 文本嵌入

        return (text_embeds, image_embeds, outputs)  # 返回结果元组

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.image_embedder 复制过来，修改了 owlvit->owlv2, OwlViTModel->Owlv2Model
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,  # 像素值
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
    ) -> Tuple[torch.FloatTensor]:
        # 获取 Owlv2Model 视觉嵌入（与 CLIP 相同）
        vision_outputs = self.owlv2.vision_model(pixel_values=pixel_values, return_dict=True)  # 获取视觉输出，返回字典格式结果

        # 将 post_layernorm 应用于 last_hidden_state，返回未投影的输出
        last_hidden_state = vision_outputs[0]  # 最后一个隐藏状态
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)  # 图像嵌入后的层归一化

        # 调整类别令牌的大小
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))  # 新尺寸
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)  # 广播类别令牌

        # 将图像嵌入与类别令牌合并
        image_embeds = image_embeds[:, 1:, :] * class_token_out  # 图像嵌入与类别令牌相乘
        image_embeds = self.layer_norm(image_embeds)  # 图像嵌入层归一化

        # 调整大小为 [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)  # 重塑图像嵌入

        return (image_embeds, vision_outputs)  # 返回结果元组

    # 从 transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.embed_image_query 复制过来
    def embed_image_query(
        self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor
    # 定义函数，接受输入并返回 torch.FloatTensor 类型的值
    ) -> torch.FloatTensor:
            # 使用类别预测器对查询图像特征进行预测，返回预测类别嵌入向量
            _, class_embeds = self.class_predictor(query_image_features)
            # 使用边界框预测器对查询图像特征和查询特征图进行预测，返回预测边界框
            pred_boxes = self.box_predictor(query_image_features, query_feature_map)
            # 将预测的边界框转换为左上角和右下角表示的格式
            pred_boxes_as_corners = center_to_corners_format(pred_boxes)
    
            # 遍历查询图像
            best_class_embeds = []
            best_box_indices = []
            pred_boxes_device = pred_boxes_as_corners.device
    
            # 循环查询图像特征的数量
            for i in range(query_image_features.shape[0]):
                # 创建每个查询框的张量，并指定设备
                each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
                each_query_pred_boxes = pred_boxes_as_corners[i]
                ious, _ = box_iou(each_query_box, each_query_pred_boxes)
    
                # 如果没有重叠的边界框，则使用广义 IoU
                if torch.all(ious[0] == 0.0):
                    ious = generalized_box_iou(each_query_box, each_query_pred_boxes)
    
                # 使用自适应阈值来包括所有与最佳 IoU 80% 相匹配的边界框
                iou_threshold = torch.max(ious) * 0.8
    
                selected_inds = (ious[0] >= iou_threshold).nonzero()
                if selected_inds.numel():
                    selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                    mean_embeds = torch.mean(class_embeds[i], axis=0)
                    mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                    best_box_ind = selected_inds[torch.argmin(mean_sim)]
                    best_class_embeds.append(class_embeds[i][best_box_ind])
                    best_box_indices.append(best_box_ind)
    
            # 如果存在最佳类别嵌入，则堆叠这些嵌入到张量中，否则设为 None
            if best_class_embeds:
                query_embeds = torch.stack(best_class_embeds)
                box_indices = torch.stack(best_box_indices)
            else:
                query_embeds, box_indices = None, None
    
            # 返回结果
            return query_embeds, box_indices, pred_boxes
    
        # 为模型前向方法添加文档字符串
        @add_start_docstrings_to_model_forward(OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2Config)
        def image_guided_detection(
            self,
            pixel_values: torch.FloatTensor,
            query_pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        # 为模型前向方法添加文档字符串
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
```