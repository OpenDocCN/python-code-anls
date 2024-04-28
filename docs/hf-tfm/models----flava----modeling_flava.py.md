# `.\models\flava\modeling_flava.py`

```
# 设置代码编码格式为 utf-8
# 版权声明和许可证信息
# 作者：Meta Platforms 和 The HuggingFace Team
# 版权所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本许可
# 除非符合许可证要求或另有书面同意，否则不得使用此文件
# 您可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律规定或书面协议，本软件的分发是在“原样”基础上进行的
# 不提供任何明示或暗示的担保或条件
# 请参阅许可证以获取有关特定语言的权限和限制
""" PyTorch FLAVA model."""

import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_flava import (
    FlavaConfig,
    FlavaImageCodebookConfig,
    FlavaImageConfig,
    FlavaMultimodalConfig,
    FlavaTextConfig,
)

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "facebook/flava-full"

# 代码本说明的检查点
_CHECKPOINT_FOR_CODEBOOK_DOC = "facebook/flava-image-codebook"
_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC = "FlavaImageConfig"
_CONFIG_CLASS_FOR_TEXT_MODEL_DOC = "FlavaTextConfig"
_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC = "FlavaMultimodalConfig"
_EXPECTED_IMAGE_OUTPUT_SHAPE = [1, 197, 768]

FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/flava-full",
    # 查看所有 FLAVA 模型 https://huggingface.co/models?filter=flava
]
FLAVA_CODEBOOK_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/flava-image-codebook"]
LOGIT_SCALE_CLAMP_MIN = 0
LOGIT_SCALE_CLAMP_MAX = 4.6052

FlavaPossibleConfigs = Union[FlavaTextConfig, FlavaImageConfig, FlavaMultimodalConfig]


@dataclass
class FlavaModelOutput(ModelOutput):
    # FLAVA 模型输出，包含来自各个编码器的嵌入和输出
    # `image_embeddings` 和 `text_embeddigns` 与从 transformer 返回的汇总输出类似
    # 如果要用于对比损失或检索的嵌入，请在 `image_embeddings` 和 `text_embeddings` 上使用 FLAVA 模型的 `image_projection` 和 `text_projection` 层
    Args:
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`].
        image_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`].
        text_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present):
            The output of the [`FlavaTextModel`].
        multimodal_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_multimodal_encoder` is `None` or `False`):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_output (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_multimodal_encoder` is `None` or `False`):
            The output of the [`FlavaMultimodalModel`].
    """

    # 定义各种类型的嵌入和输出变量，分别用于存储图像、文本和多模态的嵌入和输出结果
    image_embeddings: Optional[torch.FloatTensor] = None  # 图像嵌入向量
    image_output: Optional[BaseModelOutputWithPooling] = None  # 图像输出
    text_embeddings: Optional[torch.FloatTensor] = None  # 文本嵌入向量
    text_output: Optional[BaseModelOutputWithPooling] = None  # 文本输出
    multimodal_embeddings: Optional[torch.FloatTensor] = None  # 多模态嵌入向量
    multimodal_output: Optional[BaseModelOutputWithPooling] = None  # 多模态输出

    # 将当前对象转换为元组形式
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 遍历对象的键值对，如果键不是 "text_output", "image_output", "multimodal_output"，
            # 则直接取值，否则调用对应属性的 to_tuple 方法
            self[k] if k not in ["text_output", "image_output", "multimodal_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
from dataclasses import dataclass
from typing import Optional
from torch import ModelOutput, FloatTensor

# 定义一个数据类，表示 FLAVA 模型的预训练损失
@dataclass
class FlavaLosses(ModelOutput):
    """Class representing pretraining losses from FLAVA model

    Args:
        mim (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels` and `pixel_values` are present, `input_ids_masked` is absent and `mim_weight` > 0.:
            Masked Image Modeling loss as used in BeIT calculated only for unimodal image data.
        mlm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels` and `input_ids_masked` are present, `pixel_values` is absent and `mlm_weight` > 0.:
            Masked Language Modeling loss as used in BERT calculated only for unimodal text data.
        itm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `itm_labels`, `input_ids_masked`, `pixel_values` are present and `itm_weight` > 0.:
            Image Text Matching (ITM) loss calculated for paired image-text data. Note that ITM loss is calculated on
            masked pairs in FLAVA.
        global_contrastive (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `input_ids` and `pixel_values` are present and `global_contrastive_weight` > 0.:
            Contrastive loss for image-text similarity similar to CLIP but calculated globally for paired image-text
            data. This is calculated on unmasked images and texts.
        mmm_image (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_image_weight` > 0.:
            Masked Multimodal Modeling loss's image component calculated on paired image-text data.
        mmm_text (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_text_weight` > 0.:
            Masked Multimodal Modeling loss's text component calculated on paired image-text data.
    """

    # 定义各种预训练损失的成员变量
    mim: Optional[FloatTensor] = None
    mlm: Optional[FloatTensor] = None
    itm: Optional[FloatTensor] = None
    global_contrastive: Optional[FloatTensor] = None
    mmm_image: Optional[FloatTensor] = None
    mmm_text: Optional[FloatTensor] = None

    # 检查是否所有的成员变量都为 None
    def all_none(self) -> bool:
        all_none = True
        for v in self.values():
            if v is not None:
                all_none = False
                break
        return all_none


# 定义一个数据类，表示 FLAVA 预训练输出
@dataclass
class FlavaForPreTrainingOutput(ModelOutput):
    """
    Output from FlavaForPreTraining containing embeddings, and outputs from individual encoders.

    Note that `image_embeddings` and `text_embeddings` returned are similar to pooled output returned from a
    transformer. If you want embeddings for contrastive loss or retrieval use a FLAVA model's `image_projection` and
    `text_projection` layers on `image_embeddings` and `text_embeddings` respectively.

    """

    # 定义预训练的总损失
    loss: Optional[FloatTensor] = None
    # 定义预训练的损失信息
    loss_info: FlavaLosses = None
    # 定义图像嵌入向量
    image_embeddings: Optional[FloatTensor] = None
    # 初始化可选的图像输出结果变量，默认为 None
    image_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的文本嵌入变量，默认为 None
    text_embeddings: Optional[torch.FloatTensor] = None
    # 初始化可选的文本输出结果变量，默认为 None
    text_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的多模态嵌入变量，默认为 None
    multimodal_embeddings: Optional[torch.FloatTensor] = None
    # 初始化可选的多模态输出结果变量，默认为 None
    multimodal_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的图像掩蔽嵌入变量，默认为 None
    image_masked_embeddings: Optional[torch.FloatTensor] = None
    # 初始化可选的图像掩蔽输出结果变量，默认为 None
    image_masked_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的文本掩蔽嵌入变量，默认为 None
    text_masked_embeddings: Optional[torch.FloatTensor] = None
    # 初始化可选的文本掩蔽输出结果变量，默认为 None
    text_masked_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的多模态掩蔽嵌入变量，默认为 None
    multimodal_masked_embeddings: Optional[torch.FloatTensor] = None
    # 初始化可选的多模态掩蔽输出结果变量，默认为 None
    multimodal_masked_output: Optional[BaseModelOutputWithPooling] = None
    # 初始化可选的 MIM 损失变量，默认为 None
    mim_logits: Optional[torch.FloatTensor] = None
    # 初始化可选的 MLM 损失变量，默认为 None
    mlm_logits: Optional[torch.FloatTensor] = None
    # 初始化可选的 ITM 损失变量，默认为 None
    itm_logits: Optional[torch.FloatTensor] = None
    # 初始化可选的图像对比损失变量，默认为 None
    contrastive_logits_per_image: Optional[torch.FloatTensor] = None
    # 初始化可选的文本对比损失变量，默认为 None
    contrastive_logits_per_text: Optional[torch.FloatTensor] = None
    # 初始化可选的 MMM 图像损失变量，默认为 None
    mmm_image_logits: Optional[torch.FloatTensor] = None
    # 初始化可选的 MMM 文本损失变量，默认为 None
    mmm_text_logits: Optional[torch.FloatTensor] = None

    # 将当前对象转换为元组形式
    def to_tuple(self) -> Tuple[Any]:
        # 定义变换器输出列表
        transformer_outputs = [
            "text_output",
            "image_output",
            "multimodal_output",
            "text_masked_output",
            "image_masked_output",
            "multimodal_masked_output",
        ]
        # 返回当前对象的元组形式，如果键不在变换器输出列表中，则直接取值，否则调用 getattr 方法获取属性的元组形式
        return tuple(self[k] if k not in transformer_outputs else getattr(self, k).to_tuple() for k in self.keys())
# 基于timm实现的代码，可以在这里找到：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/image_transformer.py
# 定义一个名为FlavaImageEmbeddings的类，继承自nn.Module
class FlavaImageEmbeddings(nn.Module):
    """
    构建CLS令牌，位置和补丁嵌入。 可选择，也包括掩码令牌。
    """

    # 初始化方法，接受FlavaImageConfig对象以及use_mask_token参数，默认为False
    def __init__(self, config: FlavaImageConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 判断是否使用掩码令牌，如果use_mask_token为True或者config.mask_token有值，则为True
        use_mask_token = use_mask_token or config.mask_token
        # 初始化CLS令牌，维度为(1, 1, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 如果使用掩码令牌，则初始化掩码令牌，维度同样为(1, 1, config.hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # 初始化补丁嵌入对象，传入图片大小、补丁大小、通道数和嵌入维度等参数
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        # 计算补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 初始化位置嵌入，维度为(1, num_patches + 1, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # 初始化Dropout层，使用config中的隐藏层丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存config参数
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/image_transformer.py#L174
        """

        # 获取嵌入的维度数减1
        npatch = embeddings.shape[1] - 1
        # 获取已有的位置编码维度数减1
        num_pos = self.position_embeddings.shape[1] - 1
        # 如果嵌入维度数和位置编码维度数相同并且高度等于宽度，则直接返回位置编码
        if npatch == num_pos and height == width:
            return self.position_embeddings
        # 获取类别位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        # 获取补丁位置编码
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 获取嵌入的维度大小
        dim = embeddings.shape[-1]
        # 计算垂直方向和水平方向的补丁数量
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        # 为了避免插值中的浮点数误差，添加一个小数值
        num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1
        # 使用双三次插值方法对补丁位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(num_pos)), int(math.sqrt(num_pos)), dim).permute(0, 3, 1, 2),
            scale_factor=(num_h_patches / math.sqrt(num_pos), num_w_patches / math.sqrt(num_pos)),
            mode="bicubic",
            align_corners=False,
        )
        # 检查插值后的补丁数量是否与预期一致
        if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
            raise ValueError(
                f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
                f"shape of position embedding ({patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]})"
            )
        # 重排维度，返回插值后的位置编码
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        # 获取像素值张量的维度信息：batch_size, num_channels, height, width
        batch_size, num_channels, height, width = pixel_values.shape
        # 使用patch_embeddings函数将像素值转换成嵌入向量
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 获取嵌入向量的维度信息：batch_size, seq_len, _
        batch_size, seq_len, _ = embeddings.size()
        # 如果bool_masked_pos不为空
        if bool_masked_pos is not None:
            # 将mask_token扩展到和嵌入向量相同的维度
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将bool_masked_pos从三维转换为二维
            if bool_masked_pos.dim() == 3:
                bool_masked_pos = bool_masked_pos.view(bool_masked_pos.size(0), -1)
            # 使用mask将嵌入向量中被掩码的视觉标记替换为mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将[CLS]标记添加到嵌入的补丁标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 为每个标记添加位置编码
        if interpolate_pos_encoding:
            # 如果要插值位置编码，则添加到嵌入向量中
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，添加固定的位置编码到嵌入向量中
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量应用dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量
        return embeddings
# 基于 timm 实现的代码，可以在这里找到：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/image_transformer.py
# 定义 PatchEmbeddings 类，用于将图像转换为补丁嵌入
class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        if not isinstance(image_size, collections.abc.Iterable):
            image_size = (image_size, image_size)
        if not isinstance(patch_size, collections.abc.Iterable):
            patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用卷积层实现投影
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 投影输入像素值并展平成二维张量
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# 构建 FlavaTextEmbeddings 类，从词嵌入、位置嵌入和标记类型嵌入构建嵌入
class FlavaTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 不使用下划线，以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，长度位置嵌入）在序列化时是连续的，并在导出时被导出
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    # 定义了一个方法，用于对输入进行前向传播，生成嵌入表示
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词语 ID，可选的张量，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,  # 令牌类型 ID，可选的张量，默认为 None
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID，可选的张量，默认为 None
    ):
        # 获取输入的形状信息
        input_shape = input_ids.size()
        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置 ID 为 None，则使用预定义的位置 ID，截取到当前序列长度
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果令牌类型 ID 为 None
        if token_type_ids is None:
            # 如果模型具有 token_type_ids 属性
            if hasattr(self, "token_type_ids"):
                # 从注册的缓冲区中获取令牌类型 ID，并扩展以匹配输入的形状
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，创建全零的令牌类型 ID 张量，并使用与位置 ID 相同的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 对输入的词语进行词嵌入
        inputs_embeds = self.word_embeddings(input_ids)
        # 对令牌类型 ID 进行令牌类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和令牌类型嵌入相加得到输入的嵌入表示
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入类型为"absolute"，则加上绝对位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对嵌入表示进行 LayerNorm 规范化
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入表示进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回嵌入表示
        return embeddings
# 定义自注意力机制的类，继承自 nn.Module
class FlavaSelfAttention(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 检查隐藏层大小是否为注意力头数的整数倍，并且不包含嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化 Query、Key 和 Value 线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重排为用于计算分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的形状，去除最后一个维度，替换为 (注意力头数，每个头的大小)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重塑张量形状为新形状
        x = x.view(*new_x_shape)
        # 对张量的维度进行置换，以便后续计算
        return x.permute(0, 2, 1, 3)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 生成混合的查询向量
        mixed_query_layer = self.query(hidden_states)

        # 通过全连接层将隐藏状态转换为键向量
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 通过全连接层将隐藏状态转换为值向量
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 通过全连接层将混合的查询向量转换为查询向量
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，即查询向量与键向量的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放，以便更稳定地训练模型
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # 应用注意力掩码，将掩码应用到注意力分数上
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 再次将注意力分数归一化为概率，此处可能是冗余的
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 操作，以防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码，则将其应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与值向量的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文向量的维度进行调整，以便与原始维度匹配
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 返回模型输出，包括上下文向量和注意力分数（如果需要）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class FlavaSelfOutput(nn.Module):
    """
    The residual connection is defined in FlavaLayer (same as ViTLayer) instead of here (as is the case with other
    models), due to the layernorm applied before each block.
    """

    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，丢弃概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 通过 dropout 层处理 hidden_states
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class FlavaAttention(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 初始化 self.attention 为 FlavaSelfAttention 类的实例
        self.attention = FlavaSelfAttention(config)
        # 初始化 self.output 为 FlavaSelfOutput 类的实例
        self.output = FlavaSelfOutput(config)
        # 初始化一个空集合 pruned_heads
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 寻找可剪枝的头部和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 self.attention 处理 hidden_states
        self_outputs = self.attention(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions
        )

        # 通过 self.output 处理处理结果
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，添加 attentions
        return outputs


class FlavaIntermediate(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 初始化一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数根据 config.hidden_act 来选择
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 从 transformers.models.vit.modeling_vit.ViTIntermediate.forward 复制过来的
    # 定义一个前向传播的函数，接受隐藏状态作为输入，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态使用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class FlavaOutput(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 从transformers.models.vit.modeling_vit.ViTOutput.forward复制代码
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class FlavaLayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FlavaAttention(config)
        self.intermediate = FlavaIntermediate(config)
        self.output = FlavaOutput(config)

        # TODO: 检查fp32 layer norm的可能性
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在ViT中，self-attention之前应用layernorm
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加self注意力

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViT中，self-attention之后也应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class FlavaEncoder(nn.Module):
    def __init__(self, config: FlavaConfig) -> None:
        super().__init__()
        self.config = config
        # 创建包含指定数量FlavaLayer对象的列表
        self.layer = nn.ModuleList([FlavaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 前向传播函数，用于模型的前向计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果输出隐藏状态，则初始化一个空元组用于存储隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组用于存储注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历每个层进行前向处理
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到存储中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点并处于训练阶段，则使用梯度检查点进行前向计算
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则使用当前层的前向计算函数进行计算
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重添加到存储中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到存储中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不以字典形式返回结果，则返回所有非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 以 BaseModelOutput 类的实例形式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
class FlavaPooler(nn.Module):
    # FlavaPooler 类，继承自 nn.Module 类
    def __init__(self, config: FlavaPossibleConfigs):
        # 初始化方法，接受一个名为 config 的参数，类型为 FlavaPossibleConfigs 类
        super().__init__()
        # 调用父类的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建名为 dense 的线性层，输入和输出维度均为 config.hidden_size
        self.activation = nn.Tanh()
        # 创建激活函数为 Tanh 的实例

    def forward(self, hidden_states: torch.Tensor):
        # 前向传播方法，接受一个名为 hidden_states 的参数，类型为 torch.Tensor
        # 我们通过简单地取对应于第一个标记的隐藏状态来"池化"模型。
        first_token_tensor = hidden_states[:, 0]
        # 取出第一个标记的隐藏状态
        pooled_output = self.dense(first_token_tensor)
        # 将第一个标记的隐藏状态输入到 dense 层中
        pooled_output = self.activation(pooled_output)
        # 将 dense 层的输出输入到激活函数中
        return pooled_output
        # 返回池化后的输出

FLAVA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`{config}`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# FLAVA_START_DOCSTRING 注释
# 这是一个 PyTorch 的 torch.nn.Module 子类。将其用作常规的 PyTorch Module，并参考 PyTorch 文档了解所有与一般使用和行为相关的问题。
# 参数：
# config（[`{config}`]）：具有模型所有参数的模型配置类。
# 用配置文件初始化不会加载与模型相关的权重，只有配置。查看 `~PreTrainedModel.from_pretrained` 方法以加载模型权重。

FLAVA_INPUTS_DOCSTRING_COMMON = r"""
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

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
# FLAVA_INPUTS_DOCSTRING_COMMON 注释
# 通用输入文档字符串，包括注意力掩码、头掩码、输出注意力、输出隐藏状态和返回字典等参数说明

FLAVA_IMAGE_INPUTS_DOCSTRING_BASE = r"""
# FLAVA_IMAGE_INPUTS_DOCSTRING_BASE 注释
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。像素值可以使用 [`AutoImageProcessor`] 获得。详见 [`FlavaImageProcessor.__call__`]。

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`):
            # 布尔掩码位置。指示哪些补丁被掩码（1），哪些没有（0）。

        interpolate_pos_encoding (`bool`, *optional*):
            # 是否插值预训练位置编码。
```  
# 创建 FLAVA_IMAGE_INPUTS_DOCSTRING 字符串，它是 FLAVA_IMAGE_INPUTS_DOCSTRING_BASE 和 FLAVA_INPUTS_DOCSTRING_COMMON 的拼接
FLAVA_IMAGE_INPUTS_DOCSTRING = FLAVA_IMAGE_INPUTS_DOCSTRING_BASE + FLAVA_INPUTS_DOCSTRING_COMMON

# 创建 FLAVA_TEXT_INPUTS_DOCSTRING_BASE 字符串，它是一个带有参数占位符的原始文本输入的基础文档字符串
FLAVA_TEXT_INPUTS_DOCSTRING_BASE = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
"""

# 创建 FLAVA_TEXT_INPUTS_DOCSTRING 字符串，它是 FLAVA_TEXT_INPUTS_DOCSTRING_BASE 和 FLAVA_INPUTS_DOCSTRING_COMMON 的拼接
FLAVA_TEXT_INPUTS_DOCSTRING = FLAVA_TEXT_INPUTS_DOCSTRING_BASE + FLAVA_INPUTS_DOCSTRING_COMMON

# 创建 FLAVA_MULTIMODAL_INPUTS_DOCSTRING 字符串，它是 FLAVA_IMAGE_INPUTS_DOCSTRING_BASE、FLAVA_TEXT_INPUTS_DOCSTRING_BASE 和 FLAVA_INPUTS_DOCSTRING_COMMON 的拼接
FLAVA_MULTIMODAL_INPUTS_DOCSTRING = (
    r"""
    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, image_num_patches + text_seq_len, hidden_size)`):
            The concatenated hidden states of unimodal encoders.
"""
    + FLAVA_INPUTS_DOCSTRING_COMMON
)

# 创建 FLAVA_MODEL_INPUTS_DOCSTRING_BASE 字符串，它是模型输入文档字符串的基础部分，包含一个可选参数 skip_multimodal_encoder，用于指示是否跳过多模态编码计算
FLAVA_MODEL_INPUTS_DOCSTRING_BASE = r"""
    Args:
        skip_multimodal_encoder (*bool*, *optional*):
            Skip any calculations for multimodal encoder. Useful if multimodal encoding is not going to be used.
"""

# 创建 FLAVA_MODEL_INPUTS_DOCSTRING 字符串，它是 FLAVA_IMAGE_INPUTS_DOCSTRING_BASE、FLAVA_TEXT_INPUTS_DOCSTRING_BASE、FLAVA_INPUTS_DOCSTRING_COMMON 和 FLAVA_MODEL_INPUTS_DOCSTRING_BASE 的拼接
FLAVA_MODEL_INPUTS_DOCSTRING = (
    FLAVA_IMAGE_INPUTS_DOCSTRING_BASE
    + FLAVA_TEXT_INPUTS_DOCSTRING_BASE
    + FLAVA_INPUTS_DOCSTRING_COMMON
    + FLAVA_MODEL_INPUTS_DOCSTRING_BASE
)

# 创建 FLAVA_PRETRAINING_INPUTS_DOCSTRING 字符串，它是包含用于预训练的输入文档字符串的拼接，包括 input_ids_masked、文本输入和图像输入的基础部分
FLAVA_PRETRAINING_INPUTS_DOCSTRING = (
    r"""
    Args:
        input_ids_masked (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. These ones are the masked version of the original task
            to be used with MLM. Indices can be obtained using [`AutoTokenizer`] along with
            [`DataCollatorForMaskedLanguageModeling`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)

"""
    + FLAVA_TEXT_INPUTS_DOCSTRING_BASE
    + FLAVA_IMAGE_INPUTS_DOCSTRING_BASE
)
    r"""
        # 图像注意力遮罩，避免在图像特定的填充令牌索引上执行注意力操作。遮罩值选在 `[0, 1]` 范围内：
        # - 1 表示 **未遮罩** 的令牌，
        # - 0 表示 **遮罩** 的令牌。
        # [什么是注意力遮罩?](../glossary#attention-mask)

        # 是否跳过未遮罩的多模态编码器计算。FLAVA 预训练暂时不需要未遮罩的多模态嵌入或输出。

        # 左到右语言和多模态遮罩建模损失（下一个单词预测）的计算标签。索引应在 `[-100, 0, ..., text_config.vocab_size - 1]` 范围内
        # （参见 `input_ids` 文档字符串）。索引为 `-100` 的令牌将被忽略（遮罩），损失仅计算具有标签在 `[0, ..., text_config.vocab_size - 1]` 范围内的令牌。

        # 用于计算图像和多模态遮罩建模损失的标签。索引应在 `[-100, 0, ..., image_config.vocab_size - 1]` 范围内。索引为 `-100` 的令牌将被忽略（遮罩），
        # 仅计算具有标签在 `[0, ..., image_config.vocab_size - 1]` 范围内的令牌。如果未传入，它们将使用分配给模型的图像码书自动生成。默认情况下， 
        # 它使用 [`FlavaImageCodebook`]。查看 [`FlavaImageCodebook`] 了解如何生成 mim_labels。

        # 用于计算图像文本匹配损失的标签。0 表示不匹配，1 表示匹配。0 的对将跳过 MMM 和全局对比损失的计算。

        # 是否返回计算的损失值。

    """
"""
+ FLAVA_INPUTS_DOCSTRING_COMMON
) 

FLAVA_PRETRAINING_START_DOCSTRING_EXTRA = r"""
Parameters:
    image_codebook ([`nn.Module`]): If passed, the image codebook will be set to this. Otherwise. it will
        be initialized using the image_codebook_config defined in the config first as the first parameter.
"""


class FlavaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlavaConfig  # 设置类变量 config_class 为 FlavaConfig
    base_model_prefix = "flava"  # 设置类变量 base_model_prefix 为 "flava"
    supports_gradient_checkpointing = True  # 设置类变量 supports_gradient_checkpointing 为 True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):  # 如果 module 是 nn.Linear 或 nn.Conv2d 类型
            # 稍有不同于 TF 版本，使用标准差为 self.config.initializer_range 的正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):  # 如果 module 是 nn.Embedding 类型
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 使用标准差为 self.config.initializer_range 的正态分布初始化权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):  # 如果 module 是 nn.LayerNorm 类型
            module.bias.data.zero_()  # 将偏置初始化为 0
            module.weight.data.fill_(1.0)  # 将权重初始化为 1


@add_start_docstrings(
    "The bare FLAVA Image Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaImageConfig"),
)
class FlavaImageModel(FlavaPreTrainedModel):
    config_class = FlavaImageConfig  # 设置类变量 config_class 为 FlavaImageConfig
    # This override allows us to load FlavaImageModel from FlavaModel/FlavaForPreTraining checkpoints.
    base_model_prefix = "flava.image_model"  # 设置类变量 base_model_prefix 为 "flava.image_model"
    main_input_name = "pixel_values"  # 设置类变量 main_input_name 为 "pixel_values"

    def __init__(self, config: FlavaImageConfig, add_pooling_layer: bool = True):
        super().__init__(config)  # 调用父类的 __init__ 方法，传入参数 config

        self.config = config  # 设置实例变量 self.config 为参数 config

        self.embeddings = FlavaImageEmbeddings(config)  # 创建 FlavaImageEmbeddings 实例赋值给实例变量 self.embeddings
        self.encoder = FlavaEncoder(config)  # 创建 FlavaEncoder 实例赋值给实例变量 self.encoder

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建 nn.LayerNorm 实例赋值给 self.layernorm
        self.pooler = FlavaPooler(config) if add_pooling_layer else None  # 如果 add_pooling_layer 为 True，创建 FlavaPooler 实例赋值给 self.pooler；否则为 None

        self.post_init()  # 调用实例方法 post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embeddings  # 返回实例变量 self.embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.patch_embeddings = value  # 将参数 value 赋值给实例变量 self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():  # 遍历 heads_to_prune 字典
            self.encoder.layer[layer].attention.prune_heads(heads)  # 对指定层的注意力机制中的指定 head 进行修剪
    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    # 为模型的forward方法添加起始文档字符串，文档字符串中包含FLAVA_IMAGE_INPUTS_DOCSTRING的格式化字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC,
        modality="vision",
        expected_output=_EXPECTED_IMAGE_OUTPUT_SHAPE,
    )
    # 为代码示例添加文档字符串，包括checkpoint、output_type、config_class、modality、expected_output
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        # 设置output_attentions为给定值或者使用self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置output_hidden_states为给定值或者使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置return_dict为给定值或者使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果pixel_values为None，则抛出值错误
            raise ValueError("You have to specify pixel_values")

        # 准备头蒙版（head mask）如果需要
        # head_mask中的1.0表示保留头部
        # attention_probs的形状为 bsz x n_heads x N x N
        # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # 而head_mask会被转换成形状为[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将输入传递给嵌入层（embeddings）
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # 将嵌入输出传递给编码器（encoder）
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # 序列输出经过layernorm处理
        sequence_output = self.layernorm(sequence_output)
        # 使用池化层对序列输出进行池化，如果池化器不为None
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            # 如果不使用return_dict，返回序列输出、池化输出以及encoder_outputs中的其他部分
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 导入必要的模块和函数
@add_start_docstrings(
    "The bare FLAVA Text Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaTextConfig"),
)
# 定义 FlavaTextModel 类，继承自 FlavaPreTrainedModel
class FlavaTextModel(FlavaPreTrainedModel):
    # 设置类属性 config_class 为 FlavaTextConfig
    config_class = FlavaTextConfig
    # 设置 base_model_prefix 属性为 "flava.text_model"，用于加载 FlavaTextModel 模型
    base_model_prefix = "flava.text_model"

    # 定义初始化方法
    def __init__(self, config: FlavaTextConfig, add_pooling_layer: bool = True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将参数 config 赋值给对象的 config 属性
        self.config = config

        # 创建 FlavaTextEmbeddings 对象，并赋值给对象的 embeddings 属性
        self.embeddings = FlavaTextEmbeddings(config)
        # 创建 FlavaEncoder 对象，并赋值给对象的 encoder 属性
        self.encoder = FlavaEncoder(config)

        # 创建 LayerNorm 层，并赋值给对象的 layernorm 属性
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果 add_pooling_layer 为 True，则创建 FlavaPooler 对象，并赋值给对象的 pooler 属性；否则 pooler 属性为 None
        self.pooler = FlavaPooler(config) if add_pooling_layer else None

        # 调用对象的后初始化方法
        self.post_init()

    # 定义获取输入嵌入层的方法
    def get_input_embeddings(self) -> PatchEmbeddings:
        # 返回对象的 embeddings 属性中的 word_embeddings 属性
        return self.embeddings.word_embeddings

    # 定义设置输入嵌入层的方法
    def set_input_embeddings(self, value: nn.Module):
        # 将参数 value 赋值给对象的 embeddings 属性中的 word_embeddings 属性
        self.embeddings.word_embeddings = value

    # 定义剪枝模型头部的方法
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典
        for layer, heads in heads_to_prune.items():
            # 对模型的第 layer 层的注意力层进行头部剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_TEXT_MODEL_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        # 如果没有指定是否输出注意力权重，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有指定是否输出隐藏状态，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有指定是否返回字典格式的输出，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有输入 input_ids，则抛出 ValueError 异常
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取输入 input_ids 的形状
        input_shape = input_ids.size()

        # 如果没有提供 attention_mask，则创建一个全为 1 的 attention_mask，形状与 input_ids 相同
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)

        # 如果需要，准备头部掩码
        # 在头部掩码中，1.0 表示保留该头部的注意力
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的头部掩码的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 头部掩码转换为形状为 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 获取扩展的注意力掩码，确保形状与输入相匹配
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, input_ids.device
        )

        # 通过嵌入层处理输入数据，得到嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # 通过编码器层处理嵌入输出，得到编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出经过 LayerNormalization 处理
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化层，则对序列输出进行池化处理，得到池化输出
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不需要以字典格式返回结果，则返回元组形式的结果
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 以 BaseModelOutputWithPooling 格式返回结果
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 添加文档字符串，描述 FLAVA Multimodal Model transformer 输出原始隐藏状态而不带有特定的顶部头部
# 使用 FLAVA_START_DOCSTRING 格式化字符串填充配置信息
@add_start_docstrings(
    "The bare FLAVA Multimodal Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaMultimodalConfig"),
)
# 定义 FlavaMultimodalModel 类，继承自 FlavaPreTrainedModel
class FlavaMultimodalModel(FlavaPreTrainedModel):
    # 设置配置类为 FlavaMultimodalConfig
    config_class = FlavaMultimodalConfig
    # 允许从 FlavaModel/FlavaForPreTraining 检查点加载 FlavaMultimodalModel
    base_model_prefix = "flava.multimodal_model"
    main_input_name = "hidden_states"

    # 初始化方法
    def __init__(self, config: FlavaMultimodalConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.use_cls_token = self.config.use_cls_token
        # 如果使用 cls_token，则创建一个参数
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # 创建 FlavaEncoder 对象
        self.encoder = FlavaEncoder(config)

        # 创建 layernorm 层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果 add_pooling_layer 为真，则创建 FlavaPooler 对象，否则为 None
        self.pooler = FlavaPooler(config) if add_pooling_layer else None

        # 调用后初始化方法
        self.post_init()

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 增加模型前向方法的文档字符串
    # 使用 FLAVA_MULTIMODAL_INPUTS_DOCSTRING 格式化字符串填充输入说明
    # 使用 _CHECKPOINT_FOR_DOC、BaseModelOutputWithPooling、_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC 作为示例代码的检查点、输出类型和配置类
    @add_start_docstrings_to_model_forward(
        FLAVA_MULTIMODAL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC,
    )
    # 前向方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, BaseModelOutputWithPooling]:
        # 如果未指定输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取隐藏状态的批量大小、序列长度
        batch_size, seq_length, _ = hidden_states.size()

        # 如果使用CLS标记，扩展hidden_states并在开头添加CLS标记
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
            seq_length += 1

        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码张量
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)

        # 准备头部掩码，如果需要
        # head_mask中的1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask将转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states.device
        )

        # 编码器处理隐藏状态、注意力掩码、头部掩码等参数，返回编码器输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不返回字典，则返回元组；否则返回BaseModelOutputWithPooling对象
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
``` 
# 为 FLAVA 模型添加文档字符串
@add_start_docstrings(
    "The bare FLAVA Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaConfig"),
)
# 定义 FLAVA 模型类，继承自 FlavaPreTrainedModel
class FlavaModel(FlavaPreTrainedModel):
    # 定义配置类为 FlavaConfig
    config_class = FlavaConfig

    # 初始化函数，接受 FlavaConfig 类型的参数
    def __init__(self, config: FlavaConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查配置中的 text_config 是否为 FlavaTextConfig 类型，不是则抛出异常
        if not isinstance(config.text_config, FlavaTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type FlavaTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查配置中的 image_config 是否为 FlavaImageConfig 类型，不是则抛出异常
        if not isinstance(config.image_config, FlavaImageConfig):
            raise ValueError(
                "config.image_config is expected to be of type FlavaImageConfig but is of type"
                f" {type(config.image_config)}."
            )

        # 检查配置中的 multimodal_config 是否为 FlavaMultimodalConfig 类型，不是则抛出异常
        if not isinstance(config.multimodal_config, FlavaMultimodalConfig):
            raise ValueError(
                "config.multimodal_config is expected to be of type FlavaMultimodalConfig but "
                + f"is of type {type(config.multimodal_config)}."
            )

        # 将各个配置中的参数提取出来
        text_config = config.text_config
        image_config = config.image_config
        multimodal_config = config.multimodal_config

        # 设置特征投影的维度
        self.projection_dim = config.projection_dim
        self.text_hidden_size = text_config.hidden_size
        self.image_hidden_size = image_config.hidden_size
        self.mm_hidden_size = multimodal_config.hidden_size

        # 初始化文本模型、图片模型、多模态模型
        self.text_model = FlavaTextModel(text_config)
        self.image_model = FlavaImageModel(image_config)
        self.multimodal_model = FlavaMultimodalModel(multimodal_config)

        # 初始化图片投影和文本投影层
        self.image_projection = nn.Linear(self.image_hidden_size, self.projection_dim)
        self.text_projection = nn.Linear(self.text_hidden_size, self.projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化将图片和文本投影到多模态隐藏层的层
        self.image_to_mm_projection = nn.Linear(self.image_hidden_size, self.mm_hidden_size)
        self.text_to_mm_projection = nn.Linear(self.text_hidden_size, self.mm_hidden_size)
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    # 固定的文本特征获取函数
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token id
        attention_mask: Optional[torch.Tensor] = None,  # 注意力 mask
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 id
        position_ids: Optional[torch.Tensor] = None,  # 位置 id
        output_attentions: Optional[bool] = None,  # 是否输出 attention
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`FlavaTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("{0}")
        >>> processor = AutoProcessor.from_pretrained("{0}")

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], max_length=77, padding="max_length", return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```""".format(_CHECKPOINT_FOR_DOC)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[0]  # last_hidden_state
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    def get_image_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`FlavaImageModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("{0}")
        >>> processor = AutoProcessor.from_pretrained("{0}")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```""".format(_CHECKPOINT_FOR_DOC)
        # 使用图像模型获取图像输出
        image_outputs = self.image_model(
            pixel_values=pixel_values,  # 像素值
            bool_masked_pos=bool_masked_pos,  # 布尔类型的掩码位置
            attention_mask=attention_mask,  # 注意力掩码
            head_mask=head_mask,  # 头掩码
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            interpolate_pos_encoding=interpolate_pos_encoding,  # 是否插值位置编码
            return_dict=return_dict,  # 返回字典
        )

        # 提取图像输出的池化结果
        pooled_output = image_outputs[0]  # 最后的隐藏状态
        # 通过图像投影层得到图像特征
        image_features = self.image_projection(pooled_output)

        # 返回图像特征
        return image_features

    @add_start_docstrings_to_model_forward(
        FLAVA_MODEL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len")
    )
    @replace_return_docstrings(output_type=FlavaModelOutput, config_class=FlavaConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的标识符
        pixel_values: Optional[torch.FloatTensor] = None,  # 像素值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.Tensor] = None,  # 标记类型标识符
        bool_masked_pos: Optional[torch.Tensor] = None,  # 布尔类型的掩码位置
        position_ids: Optional[torch.LongTensor] = None,  # 位置标识符
        image_attention_mask: Optional[torch.Tensor] = None,  # 图像注意力掩码
        skip_multimodal_encoder: Optional[bool] = None,  # 是否跳过多模态编码器
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: bool = True,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
# 定义了一个名为FlavaImageCodebookResPath的类，继承自nn.Module类，表示FLAVA的图像词典模型中的ResPath组件，用于提取特征
class FlavaImageCodebookResPath(nn.Module):
    # 定义了类的初始化方法，其中in_size表示输入的通道数，out_size表示输出的通道数
    def __init__(self, in_size: int, out_size: int, **kwargs):
        super().__init__()
        hid_size = out_size // 4  # 计算隐藏层通道数，为输出通道数的1/4

        path = OrderedDict()  # 创建有序字典，用于存储各层操作
        path["relu_1"] = nn.ReLU()  # 添加ReLU激活函数层
        path["conv_1"] = nn.Conv2d(in_size, hid_size, kernel_size=3, padding=1)  # 添加卷积层1，输入通道数为in_size，输出通道数为hid_size，卷积核大小为3x3，padding为1
        path["relu_2"] = nn.ReLU()  # 添加ReLU激活函数层
        path["conv_2"] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)  # 添加卷积层2，输入通道数为hid_size，输出通道数为hid_size，卷积核大小为3x3，padding为1
        path["relu_3"] = nn.ReLU()  # 添加ReLU激活函数层
        path["conv_3"] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)  # 添加卷积层3，输入通道数为hid_size，输出通道数为hid_size，卷积核大小为3x3，padding为1
        path["relu_4"] = nn.ReLU()  # 添加ReLU激活函数层
        path["conv_4"] = nn.Conv2d(hid_size, out_size, kernel_size=1, padding=0)  # 添加卷积层4，输入通道数为hid_size，输出通道数为out_size，卷积核大小为1x1，padding为0

        self.path = nn.Sequential(path)  # 创建包含path中各层的顺序容器

    # 定义前向传播方法，其中x为输入的张量，返回特征提取后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)

# 定义了一个名为FlavaImageCodebookBlock的类，继承自nn.Module类，表示FLAVA的图像词典模型中的Block组件，用于提取特征
class FlavaImageCodebookBlock(nn.Module):
    # 定义了类的初始化方法，其中in_size表示输入的通道数，out_size表示输出的通道数，num_layers表示重复使用ResPath的次数
    def __init__(self, in_size: int, out_size: int, num_layers: int, **kwargs):
        super().__init__()

        self.post_gain = 1 / (num_layers**2)  # 计算后增益因子，用于调整残差路径的输出

        if in_size != out_size:
            self.id_path = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)  # 如果输入通道数与输出通道数不相等，则添加卷积层，通过1x1卷积改变通道数
        else:
            self.id_path = nn.Identity()  # 如果输入通道数与输出通道数相等，则添加Identity层，维度不变

        self.res_path = FlavaImageCodebookResPath(in_size, out_size)  # 创建ResPath组件实例

    # 定义前向传播方法，其中x为输入的张量，返回特征提取并融合后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

# 定义了一个名为FlavaImageCodebookLayerGroup的类，继承自nn.Module类，表示FLAVA的图像词典模型中的LayerGroup组件，用于提取特征
class FlavaImageCodebookLayerGroup(nn.Module):
    # 定义了类的初始化方法，其中num_blocks表示Block组件的个数，num_layers表示每个Block组件中的ResPath组件重复使用的次数，in_size表示输入的通道数，out_size表示输出的通道数，use_pool表示是否使用池化操作
    def __init__(self, num_blocks: int, num_layers: int, in_size: int, out_size: int, use_pool: bool = True):
        super().__init__()
        blocks = OrderedDict()  # 创建有序字典，用于存储各个Block组件

        # 使用循环依次添加Block组件
        for i in range(num_blocks):
            if i == 0:
                blocks[f"block_{i+1}"] = FlavaImageCodebookBlock(in_size, out_size, num_layers)  # 第一个Block组件，输入通道数为in_size，输出通道数为out_size
            else:
                blocks[f"block_{i+1}"] = FlavaImageCodebookBlock(out_size, out_size, num_layers)  # 其他Block组件，输入通道数与输出通道数都为out_size

        if use_pool:
            blocks["pool"] = nn.MaxPool2d(kernel_size=2)  # 添加池化层，池化核大小为2x2

        self.group = nn.Sequential(blocks)  # 创建包含blocks中各个Block组件和池化层的顺序容器

    # 定义前向传播方法，其中x为输入的张量，返回特征提取后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group(x)

# 定义了一个名为FlavaImageCodebook的类，继承自FlavaPreTrainedModel类，表示FLAVA的图像词典模型，用于生成图像特征
class FlavaImageCodebook(FlavaPreTrainedModel):
    base_model_prefix = ""  # 模型名称前缀
    config_class = FlavaImageCodebookConfig  # 模型配置类
    main_input_name = "pixel_values"  # 主要输入名称
    supports_gradient_checkpointing = False  # 是否支持梯度检查点

    # 定义了类的初始化方法，其中config表示模型的配置参数
    def __init__(
        self,
        config: FlavaImageCodebookConfig,
        **kwargs: Any,
    # 调用父类的构造函数，传入配置参数
    super().__init__(config)

    # 设置对象的配置参数
    self.config = config
    self.num_groups = config.num_groups
    self.input_channels = config.input_channels
    self.num_blocks_per_group = config.num_blocks_per_group
    self.hidden_size = config.hidden_size
    self.vocab_size = config.vocab_size

    # 计算总的层数
    num_layers = self.num_groups * self.num_blocks_per_group

    # 创建输出模块
    output_blocks = OrderedDict()
    output_blocks["relu"] = nn.ReLU()
    output_blocks["conv"] = nn.Conv2d(8 * self.hidden_size, self.vocab_size, kernel_size=1, padding=0)

    # 创建模块组
    blocks = OrderedDict()
    blocks["input"] = nn.Conv2d(self.input_channels, 1 * self.hidden_size, kernel_size=7, padding=3)
    blocks["group_1"] = FlavaImageCodebookLayerGroup(self.num_blocks_per_group, num_layers, 1 * self.hidden_size, 1 * self.hidden_size)
    blocks["group_2"] = FlavaImageCodebookLayerGroup(self.num_blocks_per_group, num_layers, 1 * self.hidden_size, 2 * self.hidden_size)
    blocks["group_3"] = FlavaImageCodebookLayerGroup(self.num_blocks_per_group, num_layers, 2 * self.hidden_size, 4 * self.hidden_size)
    blocks["group_4"] = FlavaImageCodebookLayerGroup(self.num_blocks_per_group, num_layers, 4 * self.hidden_size, 8 * self.hidden_size, use_pool=False)
    blocks["output"] = nn.Sequential(output_blocks)

    # 创建模块组的序列
    self.blocks = nn.Sequential(blocks)

    # 调用后初始化函数
    self.post_init()

    # 如果配置参数中指定需要冻结模型，则将模型参数设置为不可训练
    if self.config.freeze:
        for param in self.parameters():
            param.requires_grad = False

def get_codebook_indices(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Codebook pixel values can be obtained using [`AutoImageProcessor`] by passing
            `return_codebook_pixels=True`. See [`FlavaImageProcessor.__call__`] for details.

    Examples:
    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoImageProcessor, FlavaImageCodebook

    >>> model = FlavaImageCodebook.from_pretrained("{0}")
    >>> image_processor = AutoImageProcessor.from_pretrained("{0}")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = image_processor([image], return_codebook_pixels=True, return_tensors="pt")
    >>> inputs = dict(pixel_values=inputs.codebook_pixel_values)

    >>> outputs = model.get_codebook_indices(**inputs)
    ```
    """.format(_CHECKPOINT_FOR_CODEBOOK_DOC)
    # 通过模块组计算像素值的 logits
    z_logits = self.blocks(pixel_values)
    # 返回 logits 中最大值的索引
    return torch.argmax(z_logits, axis=1)
    # 获取代码本概率，根据像素值计算 logits
    def get_codebook_probs(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 通过模型的块处理像素值，得到 z 的 logits
        z_logits = self.blocks(pixel_values)
        # 对 logits 进行 softmax 处理，得到概率值
        return nn.Softmax(dim=1)(z_logits)

    # 前向传播函数
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                像素值。可以使用 [`AutoImageProcessor`] 通过传递 `return_codebook_pixels=True` 来获取代码本像素值。
                有关详细信息，请参阅 [`FlavaImageProcessor.__call__`]。

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoImageProcessor, FlavaImageCodebook

        >>> model = FlavaImageCodebook.from_pretrained("{0}")
        >>> image_processor = AutoImageProcessor.from_pretrained("{0}")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor([image], return_codebook_pixels=True, return_tensors="pt")
        >>> inputs = dict(pixel_values=inputs.codebook_pixel_values)

        >>> outputs = model(**inputs)
        >>> print(outputs.shape)
        (1, 196)
        ```
        """.format(_CHECKPOINT_FOR_CODEBOOK_DOC)

        # 检查输入像素值的形状是否为四维
        if len(pixel_values.shape) != 4:
            raise ValueError(f"input shape {pixel_values.shape} is not 4d")
        # 检查输入通道数是否与模型建立时的输入通道数相匹配
        if pixel_values.shape[1] != self.input_channels:
            raise ValueError(f"input has {pixel_values.shape[1]} channels but model built for {self.input_channels}")
        # 通过模型的块处理像素值
        return self.blocks(pixel_values)
# 定义一个用于预测头变换的类，继承自 nn.Module
class FlavaPredictionHeadTransform(nn.Module):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果 hidden_act 是字符串，则使用 ACT2FN 字典中对应的激活函数，否则使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 通过激活函数进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 通过 LayerNorm 进行归一化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 定义一个用于预测头的类，继承自 nn.Module
class FlavaMaskedPredictionHead(nn.Module):
    # 初始化方法，接受一个配置参数和权重参数（可选）
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        # 创建一个 FlavaPredictionHeadTransform 实例
        self.transform = FlavaPredictionHeadTransform(config)
        # 创建一个全连接层，输入维度为 config.hidden_size，输出维度为 config.vocab_size，无偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建一个偏置参数，维度为 config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 如果传入了权重参数，则使用该权重参数
        if weight is not None:
            self.decoder.weight = weight

        # 需要一个链接，以便偏置能够随着 `resize_token_embeddings` 方法的调整而正确地改变大小
        self.decoder.bias = self.bias

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 通过预测头变换对输入进行变换
        x = self.transform(x)
        # 通过全连接层进行线性变换
        x = self.decoder(x)
        return x


# 定义一个用于 ITM 头的类，继承自 nn.Module
class FlavaITMHead(nn.Module):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个 FlavaPooler 实例
        self.pooler = FlavaPooler(config)
        # 创建一个全连接层，输入维度为 config.hidden_size，输出维度为 2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 通过池化层对输入进行池化
        x = self.pooler(x)
        # 通过全连接层进行线性变换
        x = self.seq_relationship(x)
        return x


# 定义一个用于全局对比头的类，继承自 nn.Module
class FlavaGlobalContrastiveHead(nn.Module):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 设置全局反向传播对比
        self.global_backprop_contrastive = config.global_backprop_contrastive
    # 定义一个前向传播函数，接受图像嵌入、文本嵌入和 logit 缩放作为输入
    def forward(self, image_embeddings, text_embeddings, logit_scale):
        # 温度参数，用于调整对数概率的分布
        temperature = torch.exp(logit_scale)
        # 检查是否支持分布式训练且是否已初始化
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            # 生成标签，范围为图像嵌入的数量，设备与图像嵌入相同
            labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
            # 将当前图像嵌入和文本嵌入保存到列表中
            image_embeddings_all = [image_embeddings]
            text_embeddings_all = [text_embeddings]
        else:
            # 获取本地批次大小和世界大小
            local_batch_size = image_embeddings.size(0)
            world_size = torch.distributed.get_world_size()

            if self.global_backprop_contrastive:
                # 如果全局反向传播对比中的标志为真，则进行全局 all_gather 操作
                # `torch.distributed.nn.functional.all_gather` 在所有活动的工作进程上进行反向传播，
                # 而 `torch.distributed.all_gather` 只在当前工作进程上反向传播。
                image_embeddings_all = torch.distributed.nn.functional.all_gather(image_embeddings)
                text_embeddings_all = torch.distributed.nn.functional.all_gather(text_embeddings)
            else:
                # 否则，初始化图像嵌入和文本嵌入的列表，每个列表项都是与世界大小相同的张量
                image_embeddings_all = [torch.zeros_like(text_embeddings) for _ in range(world_size)]
                text_embeddings_all = [torch.zeros_like(image_embeddings) for _ in range(world_size)]
                # 在所有工作进程上进行 all_gather 操作，将结果存储在初始化的列表中
                torch.distributed.all_gather(image_embeddings_all, image_embeddings)
                torch.distributed.all_gather(text_embeddings_all, text_embeddings)

            # 生成标签，为本地批次大小乘以当前进程的排名加上范围为本地批次大小的张量，设备与图像嵌入相同
            labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
                local_batch_size, device=image_embeddings.device
            )

        # 拼接所有图像嵌入和文本嵌入
        image_embeddings_all = torch.cat(image_embeddings_all)
        text_embeddings_all = torch.cat(text_embeddings_all)

        # 计算每个图像嵌入和所有文本嵌入的点积，并乘以温度
        logits_per_image = torch.matmul(image_embeddings, text_embeddings_all.transpose(0, 1)) * temperature
        # 计算每个文本嵌入和所有图像嵌入的点积，并乘以温度
        logits_per_text = torch.matmul(text_embeddings, image_embeddings_all.transpose(0, 1)) * temperature

        # 返回图像和文本的对数概率以及标签
        return logits_per_image, logits_per_text, labels
# 添加起始文档字符串，描述 FLAVA 模型用于预训练的功能
@add_start_docstrings(
    """
    The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.
    """,
    FLAVA_START_DOCSTRING.format(config="FlavaConfig") + FLAVA_PRETRAINING_START_DOCSTRING_EXTRA,
)
# 创建 FLAVA 用于预训练的模型类
class FlavaForPreTraining(FlavaPreTrainedModel):
    # 这些与 xxx.bias 相关联
    _tied_weights_keys = [
        "mmm_text_head.decoder.bias",
        "mmm_image_head.decoder.bias",
        "mlm_head.decoder.bias",
        "mim_head.decoder.bias",
    ]

    # 初始化函数，接受 FLAVA 配置和可选的图像码本模块作为参数
    def __init__(self, config: FlavaConfig, image_codebook: Optional[nn.Module] = None):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建 FLAVA 模型
        self.flava = FlavaModel(config)

        # 设置图像码本
        self.image_codebook = image_codebook
        if self.image_codebook is None and config.init_codebook:
            self.image_codebook = FlavaImageCodebook(config.image_codebook_config)

        # 基于文本和图像编码器配置来创建遮蔽头，因为它具有正确的词汇表
        self.mim_head = FlavaMaskedPredictionHead(config.image_config)
        self.mlm_head = FlavaMaskedPredictionHead(config.text_config)
        self.itm_head = FlavaITMHead(config)
        self.mmm_image_head = FlavaMaskedPredictionHead(config.image_config)
        self.mmm_text_head = FlavaMaskedPredictionHead(config.text_config)
        self.global_contrastive_head = FlavaGlobalContrastiveHead(config)

        # 设置图像和文本词汇表大小，以及各种权重和参数
        self.image_vocab_size = config.image_config.vocab_size
        self.text_vocab_size = config.text_config.vocab_size
        self.mlm_weight = config.mlm_weight
        self.mim_weight = config.mim_weight
        self.global_contrastive_weight = config.global_contrastive_weight
        self.ce_ignore_index = config.ce_ignore_index
        self.itm_weight = config.itm_weight
        self.mmm_image_weight = config.mmm_image_weight
        self.mmm_text_weight = config.mmm_text_weight
        self.skip_unmasked_multimodal_encoder = config.skip_unmasked_multimodal_encoder

        # 调用后续初始化函数
        self.post_init()

    # 将输入张量调整为二维张量的函数
    def _resize_to_2d(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x

    # 添加前向传播函数的文档字符串
    @add_start_docstrings_to_model_forward(
        FLAVA_PRETRAINING_INPUTS_DOCSTRING.format("batch_size, text_seq_len", "batch_size, image_num_patches")
    )
    # 替换返回结果的文档字符串
    @replace_return_docstrings(output_type=FlavaForPreTrainingOutput, config_class=FlavaConfig)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        # 输入的 token IDs，类型为可选的长整型张量，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 掩码后的输入 token IDs，类型为可选的长整型张量，默认为 None
        input_ids_masked: Optional[torch.LongTensor] = None,
        # 像素值，类型为可选的浮点数张量，默认为 None
        pixel_values: Optional[torch.FloatTensor] = None,
        # 用于编码矢量量化器的像素值，类型为可选的浮点数张量，默认为 None
        codebook_pixel_values: Optional[torch.FloatTensor] = None,
        # 注意力掩码，类型为可选的张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # token 类型 IDs，类型为可选的张量，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,
        # 用于布尔掩码的位置，类型为可选的张量，默认为 None
        bool_masked_pos: Optional[torch.Tensor] = None,
        # 位置 IDs，类型为可选的长整型张量，默认为 None
        position_ids: Optional[torch.LongTensor] = None,
        # 图像注意力掩码，类型为可选的张量，默认为 None
        image_attention_mask: Optional[torch.Tensor] = None,
        # 是否跳过未掩码的多模态编码器，类型为布尔值，默认为 None
        skip_unmasked_multimodal_encoder: bool = None,
        # 用于 MLM（Masked Language Modeling）的标签，类型为可选的张量，默认为 None
        mlm_labels: Optional[torch.Tensor] = None,
        # 用于 MIM（Masked Image Modeling）的标签，类型为可选的张量，默认为 None
        mim_labels: Optional[torch.Tensor] = None,
        # 用于 ITM（Image-Text Matching）的标签，类型为可选的张量，默认为 None
        itm_labels: Optional[torch.Tensor] = None,
        # 是否输出注意力，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 True
        output_hidden_states: bool = True,
        # 是否返回字典，默认为 None
        return_dict: Optional[bool] = None,
        # 是否返回损失，默认为 None
        return_loss: Optional[bool] = None,
```