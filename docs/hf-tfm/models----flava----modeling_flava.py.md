# `.\models\flava\modeling_flava.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明和许可证信息
# 根据 Apache License 2.0 许可证，除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取完整许可证文本：http://www.apache.org/licenses/LICENSE-2.0
# 本软件基于 "AS IS" 原则发布，不提供任何形式的明示或暗示担保或条件
# 详细信息请参阅许可证文档

""" PyTorch FLAVA model. """

# 导入所需的库和模块
import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 从外部模块导入特定的函数和类
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
# 导入 FLAVA 相关配置类
from .configuration_flava import (
    FlavaConfig,
    FlavaImageCodebookConfig,
    FlavaImageConfig,
    FlavaMultimodalConfig,
    FlavaTextConfig,
)

# 获取 logger 对象用于记录日志信息
logger = logging.get_logger(__name__)

# 用于文档的预训练模型检查点名称
_CHECKPOINT_FOR_DOC = "facebook/flava-full"

# 用于图像代码簿文档的预训练模型检查点名称
_CHECKPOINT_FOR_CODEBOOK_DOC = "facebook/flava-image-codebook"
# 图像模型配置类的文档字符串
_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC = "FlavaImageConfig"
# 文本模型配置类的文档字符串
_CONFIG_CLASS_FOR_TEXT_MODEL_DOC = "FlavaTextConfig"
# 多模态模型配置类的文档字符串
_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC = "FlavaMultimodalConfig"
# 预期的图像输出形状
_EXPECTED_IMAGE_OUTPUT_SHAPE = [1, 197, 768]

# FLAVA 预训练模型的模型存档列表
FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/flava-full",
    # 可以在 https://huggingface.co/models?filter=flava 查看所有 FLAVA 模型
]
# FLAVA 图像代码簿预训练模型的模型存档列表
FLAVA_CODEBOOK_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/flava-image-codebook"]
# 对数尺度的最小值
LOGIT_SCALE_CLAMP_MIN = 0
# 对数尺度的最大值
LOGIT_SCALE_CLAMP_MAX = 4.6052

# FLAVA 模型可能的配置类别
FlavaPossibleConfigs = Union[FlavaTextConfig, FlavaImageConfig, FlavaMultimodalConfig]

# 数据类，包含 FLAVA 模型的输出，继承自 ModelOutput 类
@dataclass
class FlavaModelOutput(ModelOutput):
    """
    FlavaModel 的输出，包含来自各个编码器的嵌入和输出。

    注意，返回的 `image_embeddings` 和 `text_embeddings` 类似于变压器返回的汇总输出。
    如果需要用于对比损失或检索的嵌入，请在 `image_embeddings` 和 `text_embeddings` 上使用 FLAVA 模型的
    `image_projection` 和 `text_projection` 层。
    """
    """
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

    # 可选的图像嵌入向量，形状为 `(batch_size, output_dim)`，当存在 `pixel_values` 时返回
    image_embeddings: Optional[torch.FloatTensor] = None
    # 可选的图像模型输出，当存在 `pixel_values` 时返回
    image_output: Optional[BaseModelOutputWithPooling] = None
    # 可选的文本嵌入向量，形状为 `(batch_size, output_dim)`，当存在 `input_ids` 时返回
    text_embeddings: Optional[torch.FloatTensor] = None
    # 可选的文本模型输出，当存在 `input_ids` 时返回
    text_output: Optional[BaseModelOutputWithPooling] = None
    # 可选的多模态嵌入向量，形状为 `(batch_size, output_dim)`，当同时存在 `input_ids` 和 `pixel_values` 并且 `skip_multimodal_encoder` 不为 `None` 或 `False` 时返回
    multimodal_embeddings: Optional[torch.FloatTensor] = None
    # 可选的多模态模型输出，当同时存在 `input_ids` 和 `pixel_values` 并且 `skip_multimodal_encoder` 不为 `None` 或 `False` 时返回
    multimodal_output: Optional[BaseModelOutputWithPooling] = None

    # 将当前对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 对于所有键，返回对应的值，除非键是 ["text_output", "image_output", "multimodal_output"] 中的一个，
            # 这些键对应的值需调用其 `to_tuple()` 方法进行转换
            self[k] if k not in ["text_output", "image_output", "multimodal_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 定义一个数据类 `FlavaLosses`，用于存储 FLAVA 模型的预训练损失
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

    # 定义各种损失的属性，使用 torch.FloatTensor 类型，均可选
    mim: Optional[torch.FloatTensor] = None
    mlm: Optional[torch.FloatTensor] = None
    itm: Optional[torch.FloatTensor] = None
    global_contrastive: Optional[torch.FloatTensor] = None
    mmm_image: Optional[torch.FloatTensor] = None
    mmm_text: Optional[torch.FloatTensor] = None

    # 定义一个方法，用于检查所有损失属性是否都为 None
    def all_none(self) -> bool:
        # 初始化一个标志位，表示是否所有属性都为 None
        all_none = True
        # 遍历所有损失属性的值
        for v in self.values():
            # 如果某个属性值不为 None，则将标志位置为 False，并跳出循环
            if v is not None:
                all_none = False
                break
        # 返回标志位，表示所有损失属性是否都为 None
        return all_none


# 定义一个数据类 `FlavaForPreTrainingOutput`，用于存储 FLAVA 模型预训练的输出
@dataclass
class FlavaForPreTrainingOutput(ModelOutput):
    """
    Output from FlavaForPreTraining containing embeddings, and outputs from individual encoders.

    Note that `image_embeddings` and `text_embeddings` returned are similar to pooled output returned from a
    transformer. If you want embeddings for contrastive loss or retrieval use a FLAVA model's `image_projection` and
    `text_projection` layers on `image_embeddings` and `text_embeddings` respectively.

    """

    # 定义模型输出的属性，包括损失、损失信息以及图像嵌入
    loss: Optional[torch.FloatTensor] = None
    loss_info: FlavaLosses = None
    image_embeddings: Optional[torch.FloatTensor] = None
    # 定义多个可选的模型输出变量，初始值均为 None
    image_output: Optional[BaseModelOutputWithPooling] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    text_output: Optional[BaseModelOutputWithPooling] = None
    multimodal_embeddings: Optional[torch.FloatTensor] = None
    multimodal_output: Optional[BaseModelOutputWithPooling] = None
    image_masked_embeddings: Optional[torch.FloatTensor] = None
    image_masked_output: Optional[BaseModelOutputWithPooling] = None
    text_masked_embeddings: Optional[torch.FloatTensor] = None
    text_masked_output: Optional[BaseModelOutputWithPooling] = None
    multimodal_masked_embeddings: Optional[torch.FloatTensor] = None
    multimodal_masked_output: Optional[BaseModelOutputWithPooling] = None
    mim_logits: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    itm_logits: Optional[torch.FloatTensor] = None
    contrastive_logits_per_image: Optional[torch.FloatTensor] = None
    contrastive_logits_per_text: Optional[torch.FloatTensor] = None
    mmm_image_logits: Optional[torch.FloatTensor] = None
    mmm_text_logits: Optional[torch.FloatTensor] = None

    # 定义方法将对象转换为元组的函数签名
    def to_tuple(self) -> Tuple[Any]:
        # 指定转换输出的顺序列表
        transformer_outputs = [
            "text_output",
            "image_output",
            "multimodal_output",
            "text_masked_output",
            "image_masked_output",
            "multimodal_masked_output",
        ]
        # 返回一个元组，包含对象中指定的属性的值，若属性在 transformer_outputs 中，则调用相应对象的 to_tuple() 方法
        return tuple(self[k] if k not in transformer_outputs else getattr(self, k).to_tuple() for k in self.keys())
# 基于 timm 实现的代码，可以在以下链接找到：
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/image_transformer.py
class FlavaImageEmbeddings(nn.Module):
    """
    构建 CLS token、位置和patch embeddings。可选择是否包含 mask token。
    """

    def __init__(self, config: FlavaImageConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 确定是否使用 mask token，如果 use_mask_token 为 True 或者 config 中指定了 mask_token，则使用
        use_mask_token = use_mask_token or config.mask_token
        
        # 定义 CLS token，是一个可学习的参数，形状为 (1, 1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # 如果使用 mask token，则定义一个可学习的参数作为 mask token，形状同上
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        
        # 初始化 patch embeddings，使用 PatchEmbeddings 类生成 patch embeddings
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        
        # 计算 patch 的数量（加上一个额外的位置用于 CLS token），用于定义位置 embeddings
        num_patches = self.patch_embeddings.num_patches
        
        # 定义位置 embeddings，是一个可学习的参数，形状为 (1, num_patches + 1, hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        
        # 定义 dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 保存配置信息
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/image_transformer.py#L174
        """

        # 计算当前嵌入的图像块数目（npatch）
        npatch = embeddings.shape[1] - 1
        # 获取预训练位置编码的数目（num_pos）
        num_pos = self.position_embeddings.shape[1] - 1
        # 如果图像块数目与位置编码数目相等，并且图像是正方形，则直接返回位置编码
        if npatch == num_pos and height == width:
            return self.position_embeddings
        
        # 获取类别位置编码（第一列）
        class_pos_embed = self.position_embeddings[:, 0]
        # 获取图像块位置编码（除去第一列）
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 获取嵌入的维度
        dim = embeddings.shape[-1]
        # 计算图像中的水平和垂直图块数目
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        # 添加一个小数以避免插值时的浮点数误差
        num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1
        
        # 对图像块位置编码进行插值操作
        patch_pos_embed = nn.functional.interpolate(
            # 将位置编码重新形状为 4 维张量，并重新排列维度顺序
            patch_pos_embed.reshape(1, int(math.sqrt(num_pos)), int(math.sqrt(num_pos)), dim).permute(0, 3, 1, 2),
            # 设置插值的比例因子，根据图像块数目和位置编码数目的关系
            scale_factor=(num_h_patches / math.sqrt(num_pos), num_w_patches / math.sqrt(num_pos)),
            mode="bicubic",  # 使用双三次插值模式
            align_corners=False,  # 不对齐角落像素
        )
        
        # 检查插值后的图像块位置编码是否与预期的图像块数目相符
        if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
            raise ValueError(
                f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
                f"shape of position embedding ({patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]})"
            )
        
        # 调整形状并排列维度以匹配模型输出的要求
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # 将类别位置编码与插值后的图像块位置编码拼接在一起
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    # 定义一个方法，接受像素值作为输入，返回一个 Torch 张量
    ) -> torch.Tensor:
        # 获取输入张量的维度信息：批大小、通道数、高度、宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 将像素值传入 patch_embeddings 方法，生成嵌入表示，并可能插入位置编码
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 再次获取嵌入张量的维度信息：批大小、序列长度、嵌入维度
        batch_size, seq_len, _ = embeddings.size()
        # 如果存在布尔类型的遮罩位置信息
        if bool_masked_pos is not None:
            # 创建一个与嵌入张量形状相同的 mask_tokens 张量，用于替换遮罩的视觉标记
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 如果 bool_masked_pos 是三维的，将其展平为二维的
            if bool_masked_pos.dim() == 3:
                bool_masked_pos = bool_masked_pos.view(bool_masked_pos.size(0), -1)
            # 将 bool_masked_pos 转换为与 mask_tokens 相同类型的张量，并将遮罩应用到 embeddings
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将 [CLS] 标记添加到嵌入的补丁标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 如果选择插值位置编码，则对每个 token 添加位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，直接添加预定义的位置编码
            embeddings = embeddings + self.position_embeddings

        # 应用 dropout 操作到嵌入张量
        embeddings = self.dropout(embeddings)

        # 返回嵌入张量作为输出
        return embeddings
# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/image_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        # 如果image_size不是可迭代对象，则转换为元组
        if not isinstance(image_size, collections.abc.Iterable):
            image_size = (image_size, image_size)
        # 如果patch_size不是可迭代对象，则转换为元组
        if not isinstance(patch_size, collections.abc.Iterable):
            patch_size = (patch_size, patch_size)
        # 计算图像被划分成的块数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用卷积层将输入图像的每个patch映射到embed_dim维度的特征空间
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果不需要插值位置编码，检查输入图像尺寸是否与预期尺寸匹配
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 使用卷积层对输入图像进行特征提取，并展平和转置维度以适应后续处理
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class FlavaTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 创建词嵌入，词位置嵌入和令牌类型嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 保持变量名与TensorFlow模型的一致性，并且能够加载TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，绝对还是相对
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置ID张量，用于序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册令牌类型ID张量，初始化为全零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    # 定义一个方法，用于模型的前向传播，接受输入的张量和位置信息
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        # 获取输入张量的形状信息
        input_shape = input_ids.size()
        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供位置信息，则使用预设的位置信息
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 设置 token_type_ids 为构造函数中注册的缓冲区，通常为全零，用于在不传递 token_type_ids 时
        # 跟踪模型时帮助用户，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 使用预设的 token_type_ids，扩展到与输入形状相同的尺寸
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果未定义 token_type_ids，则创建全零张量，设备与位置信息相同
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 将输入的 token IDs 转换为词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取对应的 token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入向量和 token type embeddings 相加得到最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型为 "absolute"，则添加绝对位置嵌入向量
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对 embeddings 进行 Layer Normalization 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理，用于模型的正则化
        embeddings = self.dropout(embeddings)

        # 返回最终的 embeddings 作为前向传播的输出
        return embeddings
# 定义自注意力机制的模型类，继承自 nn.Module
class FlavaSelfAttention(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是且没有嵌入大小的属性，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化用于 dropout 的层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量转换为分数矩阵的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义前向传播函数，接受隐藏状态、注意力掩码、头掩码和输出注意力标志
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # Compute mixed query layer using the query projection layer
        mixed_query_layer = self.query(hidden_states)

        # Compute key and value layers by applying projection layers to hidden states
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Compute attention scores by taking the dot product of query and key tensors
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Scale the attention scores by the square root of the head size
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the provided attention mask to the attention scores
            attention_scores = attention_scores + attention_mask

        # Compute attention probabilities by applying softmax to the attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)

        # Apply head mask to attention probabilities if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Compute the context layer by taking the weighted sum of value tensors
        context_layer = torch.matmul(attention_probs, value_layer)

        # Transpose and reshape the context layer to match the required output shape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Prepare outputs depending on whether to include attention probabilities
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # Return the computed outputs
        return outputs
class FlavaSelfOutput(nn.Module):
    """
    The residual connection is defined in FlavaLayer (same as ViTLayer) instead of here (as is the case with other
    models), due to the layernorm applied before each block.
    """

    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 线性层，输入输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 以概率 config.hidden_dropout_prob 进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 随机失活
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class FlavaAttention(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 自注意力机制模块
        self.attention = FlavaSelfAttention(config)
        # 输出模块，包括线性变换和随机失活
        self.output = FlavaSelfOutput(config)
        # 被剪枝的注意力头部集合
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头部并获取索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 对线性层进行剪枝
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部
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
        # 执行自注意力机制，并返回输出
        self_outputs = self.attention(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions
        )

        # 将自注意力的输出传递给输出模块，得到最终的注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力权重，则添加到输出中
        return outputs


class FlavaIntermediate(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 线性层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则使用预定义的激活函数；否则使用配置中定义的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 从 transformers.models.vit.modeling_vit.ViTIntermediate.forward 复制过来的
    # 定义一个方法，用于前向传播计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用激活函数（通常是ReLU或类似的函数）
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过线性变换和激活函数处理后的隐藏状态
        return hidden_states
class FlavaOutput(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 创建一个线性层，用于从中间大小映射到隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个用于随机失活的层，根据隐藏失活概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 从transformers.models.vit.modeling_vit.ViTOutput.forward复制过来
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层传播隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用随机失活到传播后的隐藏状态
        hidden_states = self.dropout(hidden_states)

        # 将传播后的隐藏状态与输入张量相加
        hidden_states = hidden_states + input_tensor

        return hidden_states


class FlavaLayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: FlavaPossibleConfigs) -> None:
        super().__init__()
        # 设置用于前馈传递的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = FlavaAttention(config)
        # 初始化中间层
        self.intermediate = FlavaIntermediate(config)
        # 初始化输出层
        self.output = FlavaOutput(config)

        # TODO: 检查是否可能使用fp32层归一化
        # 在隐藏大小上使用层归一化，设置epsilon为config中的层归一化epsilon
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在自注意力之前应用层归一化，在ViT中，在自注意力之前应用层归一化
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViT中，也在自注意力之后应用层归一化
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
        # 创建一个由FlavaLayer组成的层列表，列表长度为config中的隐藏层数量
        self.layer = nn.ModuleList([FlavaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 定义一个前向传播方法，用于处理模型的前向推断过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化一个空元组来存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组来存储所有自注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且处于训练模式，则通过梯度检查点函数执行当前层的调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到所有自注意力权重元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 最后一个层完成后，如果需要输出隐藏状态，则将最终的隐藏状态加入到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则按需返回隐藏状态、所有隐藏状态、所有自注意力权重的元组形式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和所有自注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
class FlavaPooler(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        # 通过简单地选取第一个 token 对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 将选取的隐藏状态输入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 将全连接层的输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


FLAVA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`{config}`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

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

FLAVA_IMAGE_INPUTS_DOCSTRING_BASE = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values of the input images. This tensor represents the image data with dimensions:
            - `batch_size`: Number of images in the batch.
            - `num_channels`: Number of color channels (e.g., 3 for RGB images).
            - `height`: Height of each image.
            - `width`: Width of each image.
            Pixel values can be obtained using an `AutoImageProcessor`. Refer to the documentation
            of [`FlavaImageProcessor.__call__`] for more details.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`):
            Boolean tensor indicating masked positions within each image. Each element:
            - `1`: Indicates the corresponding image patch is masked.
            - `0`: Indicates the corresponding image patch is not masked.

        interpolate_pos_encoding (`bool`, *optional*):
            Optional flag indicating whether to interpolate pre-trained position encodings. If set to `True`,
            the model will interpolate existing position encodings; if `False` or not provided, no interpolation
            will be performed.
"""

FLAVA_IMAGE_INPUTS_DOCSTRING = FLAVA_IMAGE_INPUTS_DOCSTRING_BASE + FLAVA_INPUTS_DOCSTRING_COMMON
# 将基础的图像输入文档字符串与通用输入文档字符串相结合，形成完整的图像输入文档字符串

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
# 文本输入基础文档字符串，包含关于输入IDs和token type IDs的详细说明

FLAVA_TEXT_INPUTS_DOCSTRING = FLAVA_TEXT_INPUTS_DOCSTRING_BASE + FLAVA_INPUTS_DOCSTRING_COMMON
# 将基础的文本输入文档字符串与通用输入文档字符串相结合，形成完整的文本输入文档字符串

FLAVA_MULTIMODAL_INPUTS_DOCSTRING = (
    r"""
    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, image_num_patches + text_seq_len, hidden_size)`):
            The concatenated hidden states of unimodal encoders.
"""
    + FLAVA_INPUTS_DOCSTRING_COMMON
)
# 多模态输入文档字符串，描述了隐藏状态的拼接表示以及通用输入信息

FLAVA_MODEL_INPUTS_DOCSTRING_BASE = r"""
    Args:
        skip_multimodal_encoder (*bool*, *optional*):
            Skip any calculations for multimodal encoder. Useful if multimodal encoding is not going to be used.
"""
# 模型输入基础文档字符串，描述了是否跳过多模态编码器的计算的可选参数

FLAVA_MODEL_INPUTS_DOCSTRING = (
    FLAVA_IMAGE_INPUTS_DOCSTRING_BASE
    + FLAVA_TEXT_INPUTS_DOCSTRING_BASE
    + FLAVA_INPUTS_DOCSTRING_COMMON
    + FLAVA_MODEL_INPUTS_DOCSTRING_BASE
)
# 模型输入文档字符串，包含了图像、文本、和通用输入的详细说明以及模型输入基础的描述

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
# 预训练输入文档字符串，描述了用于掩码语言建模的输入IDs以及文本和图像输入的基础描述
    # 定义参数 `image_attention_mask`，用于指定哪些图像注意力应避免，避免在填充标记上执行注意力。
    # 该参数是一个形状为 `{1}` 的 PyTorch 浮点张量，可选参数。
    # 值范围为 `[0, 1]`：
    # - 1 表示 **未屏蔽** 的标记，
    # - 0 表示 **已屏蔽** 的标记。
    # 详细了解注意力遮罩，请参阅 glossary 中的 "attention-mask" 部分。

    # 跳过未屏蔽的多模态编码器，用于 FLAVA 预训练，当前不需要未屏蔽的多模态嵌入或输出。

    # 定义参数 `mlm_labels`，用于计算左到右语言和多模态屏蔽建模损失（下一个词预测）的标签。
    # 该参数是一个形状为 `(batch_size, text_seq_len)` 的 PyTorch 长整型张量，可选参数。
    # 索引应在 `[-100, 0, ..., text_config.vocab_size - 1]` 范围内（参见 `input_ids` 的文档字符串）。
    # 索引设置为 `-100` 的标记被忽略（屏蔽），仅为标签在 `[0, ..., text_config.vocab_size - 1]` 范围内的标记计算损失。

    # 定义参数 `mim_labels`，用于计算图像和多模态屏蔽建模损失的标签。
    # 该参数是一个形状为 `(batch_size, image_num_patches)` 的 PyTorch 长整型张量，可选参数。
    # 索引应在 `[-100, 0, ..., image_config.vocab_size - 1]` 范围内。
    # 索引设置为 `-100` 的标记被忽略（屏蔽），仅为标签在 `[0, ..., image_config.vocab_size - 1]` 范围内的标记计算损失。
    # 如果未传入该参数，则会自动生成，使用模型分配的图像码本。默认使用 [`FlavaImageCodebook`]。详细了解 `FlavaImageCodebook` 以了解如何生成 `mim_labels`。

    # 定义参数 `itm_labels`，用于计算图像-文本匹配损失的标签。
    # 该参数是一个形状为 `(batch_size, 1)` 的 PyTorch 长整型张量，可选参数。
    # 值 `0` 表示不匹配的对，值 `1` 表示匹配的对。
    # 值为 `0` 的对将被跳过计算 MMM 和全局对比损失。

    # 定义参数 `return_loss`，指示是否返回计算的损失。
    # 该参数是一个布尔值，可选参数，默认为 `None`。
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

    config_class = FlavaConfig
    base_model_prefix = "flava"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对于线性层和卷积层，使用正态分布初始化权重
            # 略微不同于 TF 版本，后者使用截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果指定了 padding_idx，则将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，初始化偏置为零，初始化权重为全1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare FLAVA Image Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaImageConfig"),
)
class FlavaImageModel(FlavaPreTrainedModel):
    config_class = FlavaImageConfig
    # This override allows us to load FlavaImageModel from FlavaModel/FlavaForPreTraining checkpoints.
    base_model_prefix = "flava.image_model"
    main_input_name = "pixel_values"

    def __init__(self, config: FlavaImageConfig, add_pooling_layer: bool = True):
        super().__init__(config)

        self.config = config

        # 初始化模型的各个部分
        self.embeddings = FlavaImageEmbeddings(config)
        self.encoder = FlavaEncoder(config)

        # 初始化 LayerNorm 和 Pooler（如果需要）
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = FlavaPooler(config) if add_pooling_layer else None

        # 执行初始化后的操作
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.patch_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 裁剪模型中的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)
    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC,
        modality="vision",
        expected_output=_EXPECTED_IMAGE_OUTPUT_SHAPE,
    )
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
        # 设置输出注意事项，默认为模型配置中的输出设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的输出设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果未提供像素值，则抛出数值错误异常
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值传入嵌入层进行编码
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # 将编码后的数据传入编码器进行处理
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出经过 LayerNormalization 处理
        sequence_output = self.layernorm(sequence_output)
        # 如果有池化层，对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            # 如果不返回字典，则返回元组格式的输出
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典格式的输出，则构造 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器添加文档字符串，描述这是一个在顶部没有特定头部的原始隐藏状态输出的 FLAVA 文本模型转换器。
# 使用 FLAVA_START_DOCSTRING 格式化字符串，填充 FlavaTextConfig 相关信息。
@add_start_docstrings(
    "The bare FLAVA Text Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaTextConfig"),
)
# 定义 FlavaTextModel 类，继承自 FlavaPreTrainedModel 类
class FlavaTextModel(FlavaPreTrainedModel):
    # 指定配置类为 FlavaTextConfig
    config_class = FlavaTextConfig
    # 模型前缀用于加载 FlavaTextModel 的检查点
    base_model_prefix = "flava.text_model"

    def __init__(self, config: FlavaTextConfig, add_pooling_layer: bool = True):
        # 调用父类的构造方法，传入配置对象
        super().__init__(config)
        # 保存配置对象
        self.config = config

        # 初始化嵌入层对象
        self.embeddings = FlavaTextEmbeddings(config)
        # 初始化编码器对象
        self.encoder = FlavaEncoder(config)

        # 初始化层归一化层，使用配置中的隐藏层大小和层归一化的 epsilon 参数
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果设置了添加池化层，则初始化池化层对象
        self.pooler = FlavaPooler(config) if add_pooling_layer else None

        # 执行初始化后的处理
        self.post_init()

    # 获取输入嵌入的方法，返回词嵌入层对象
    def get_input_embeddings(self) -> PatchEmbeddings:
        return self.embeddings.word_embeddings

    # 设置输入嵌入的方法，设置词嵌入层对象为指定的值
    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    # 对模型的注意力头进行修剪的方法
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 使用装饰器添加模型正向传播的文档字符串，描述输入参数的含义
    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    # 使用示例代码的文档字符串样本，描述模型的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_TEXT_MODEL_DOC,
    )
    # 模型的正向传播方法定义，接收多个输入参数，返回模型输出
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
        # 如果 output_attentions 参数为 None，则使用配置中的 output_attentions 参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数为 None，则使用配置中的 output_hidden_states 参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数为 None，则使用配置中的 use_return_dict 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 input_ids 为空，则抛出数值错误异常
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取 input_ids 的形状
        input_shape = input_ids.size()

        # 如果 attention_mask 为空，则创建全 1 的注意力掩码，形状与 input_ids 相同
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)

        # 准备头部掩码（head mask），如果需要的话
        # head_mask 中的 1.0 表示保留对应的头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # 获取扩展的注意力掩码（extended_attention_mask）
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, input_ids.device
        )

        # 通过 embeddings 模块生成嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # 使用 encoder 模块处理嵌入输出，得到编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出（sequence_output）
        sequence_output = encoder_outputs[0]
        # 应用 layernorm 层到序列输出上
        sequence_output = self.layernorm(sequence_output)
        # 如果有池化器（pooler），则应用池化器到序列输出上，得到池化输出（pooled_output）
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不需要返回字典形式的结果，则返回一个元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 否则，返回一个 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 添加文档字符串描述 FLAVA Multimodal Model 类的基本信息和配置格式
@add_start_docstrings(
    "The bare FLAVA Multimodal Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaMultimodalConfig"),
)
class FlavaMultimodalModel(FlavaPreTrainedModel):
    # 指定该类使用的配置类为 FlavaMultimodalConfig
    config_class = FlavaMultimodalConfig
    # 定义在加载模型时从 FlavaModel/FlavaForPreTraining 检查点中读取的基础模型前缀
    base_model_prefix = "flava.multimodal_model"
    # 主输入名称为 "hidden_states"
    main_input_name = "hidden_states"

    def __init__(self, config: FlavaMultimodalConfig, add_pooling_layer=True):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        self.config = config
        # 根据配置决定是否使用类别标记 (CLS token)
        self.use_cls_token = self.config.use_cls_token
        if self.use_cls_token:
            # 如果使用类别标记，则初始化一个可学习的张量作为类别标记
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # 初始化编码器，使用 FlavaEncoder 类
        self.encoder = FlavaEncoder(config)

        # 初始化层归一化层，使用指定的层归一化尺寸和 epsilon 值
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 根据参数决定是否添加池化层，如果需要则使用 FlavaPooler 类初始化池化层
        self.pooler = FlavaPooler(config) if add_pooling_layer else None

        # 调用后初始化方法，用于子类中进一步初始化操作
        self.post_init()

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头信息，通过调用编码器中的注意力层进行剪枝
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        FLAVA_MULTIMODAL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC,
    )
    # 定义模型的前向传播函数，接收输入的隐藏状态和可选的掩码和掩码头，返回模型输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 前向传播函数的文档字符串描述了输入参数的含义和模型预期的输出

        # 输入的隐藏状态张量
        self,
        # 可选的注意力掩码张量，用于控制哪些位置需要被忽略
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的头掩码张量，用于控制哪些注意力头需要被忽略
        head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否以字典格式返回结果
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, BaseModelOutputWithPooling]:
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否使用返回字典格式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取隐藏状态张量的维度
        batch_size, seq_length, _ = hidden_states.size()

        # 如果使用CLS token，则扩展并拼接隐藏状态
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
            seq_length += 1

        # 如果未提供注意力掩码，则创建全1的注意力掩码张量
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)

        # 准备头部掩码（如果需要）
        # head_mask中的1.0表示保留对应的头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # 获取扩展的注意力掩码张量
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states.device
        )

        # 将隐藏状态输入编码器
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，则对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不需要返回字典，则返回一个元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典，则构造BaseModelOutputWithPooling对象返回
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器为 FLAVA 模型类添加文档字符串，描述该模型仅输出原始隐藏状态，没有额外的顶层头部。
# FLAVA_START_DOCSTRING 包含一个格式字符串，用于填充 FlavaConfig 的信息。
@add_start_docstrings(
    "The bare FLAVA Model transformer outputting raw hidden-states without any specific head on top.",
    FLAVA_START_DOCSTRING.format(config="FlavaConfig"),
)
class FlavaModel(FlavaPreTrainedModel):
    # 指定配置类为 FlavaConfig
    config_class = FlavaConfig

    def __init__(self, config: FlavaConfig):
        # 调用父类构造函数，初始化模型
        super().__init__(config)

        # 验证文本配置是否为 FlavaTextConfig 类型，否则引发 ValueError
        if not isinstance(config.text_config, FlavaTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type FlavaTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 验证图像配置是否为 FlavaImageConfig 类型，否则引发 ValueError
        if not isinstance(config.image_config, FlavaImageConfig):
            raise ValueError(
                "config.image_config is expected to be of type FlavaImageConfig but is of type"
                f" {type(config.image_config)}."
            )

        # 验证多模态配置是否为 FlavaMultimodalConfig 类型，否则引发 ValueError
        if not isinstance(config.multimodal_config, FlavaMultimodalConfig):
            raise ValueError(
                "config.multimodal_config is expected to be of type FlavaMultimodalConfig but "
                + f"is of type {type(config.multimodal_config)}."
            )

        # 将各配置对象存储为类属性
        text_config = config.text_config
        image_config = config.image_config
        multimodal_config = config.multimodal_config

        # 初始化投影维度、文本隐藏层大小、图像隐藏层大小、多模态隐藏层大小
        self.projection_dim = config.projection_dim
        self.text_hidden_size = text_config.hidden_size
        self.image_hidden_size = image_config.hidden_size
        self.mm_hidden_size = multimodal_config.hidden_size

        # 初始化文本模型、图像模型和多模态模型
        self.text_model = FlavaTextModel(text_config)
        self.image_model = FlavaImageModel(image_config)
        self.multimodal_model = FlavaMultimodalModel(multimodal_config)

        # 初始化图像到投影空间的线性层、文本到投影空间的线性层、logit 缩放参数
        self.image_projection = nn.Linear(self.image_hidden_size, self.projection_dim)
        self.text_projection = nn.Linear(self.text_hidden_size, self.projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化图像到多模态投影空间的线性层、文本到多模态投影空间的线性层
        self.image_to_mm_projection = nn.Linear(self.image_hidden_size, self.mm_hidden_size)
        self.text_to_mm_projection = nn.Linear(self.text_hidden_size, self.mm_hidden_size)

        # 执行初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 get_text_features 方法添加文档字符串，描述该方法接受文本输入并返回相关特征
    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
    # 使用预训练模型生成文本特征，通过传入的参数组成输入
    text_outputs = self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    # 从文本输出中获取池化后的特征向量（通常是最后一个隐藏状态）
    pooled_output = text_outputs[0]  # last_hidden_state
    # 将池化后的特征向量投影到最终的文本特征空间
    text_features = self.text_projection(pooled_output)
    
    # 返回文本特征向量
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
        # 调用图像模型，传入像素值、是否遮蔽位置、注意力掩码、头掩码等参数进行推理
        image_outputs = self.image_model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # 从图像模型输出中取出汇总输出（通常是最后一个隐藏状态）
        pooled_output = image_outputs[0]  # last_hidden_state
        # 将汇总输出应用于图像投影层，生成图像特征向量
        image_features = self.image_projection(pooled_output)

        # 返回图像特征向量作为模型前向传播的结果
        return image_features

    @add_start_docstrings_to_model_forward(
        FLAVA_MODEL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len")
    )
    @replace_return_docstrings(output_type=FlavaModelOutput, config_class=FlavaConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        skip_multimodal_encoder: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: bool = True,
        return_dict: Optional[bool] = None,
class FlavaImageCodebookResPath(nn.Module):
    def __init__(self, in_size: int, out_size: int, **kwargs):
        super().__init__()
        hid_size = out_size // 4

        # 定义一个有序字典，用于存储网络的层次结构
        path = OrderedDict()
        path["relu_1"] = nn.ReLU()  # 第一个 ReLU 激活函数
        path["conv_1"] = nn.Conv2d(in_size, hid_size, kernel_size=3, padding=1)  # 第一个卷积层
        path["relu_2"] = nn.ReLU()  # 第二个 ReLU 激活函数
        path["conv_2"] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)  # 第二个卷积层
        path["relu_3"] = nn.ReLU()  # 第三个 ReLU 激活函数
        path["conv_3"] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)  # 第三个卷积层
        path["relu_4"] = nn.ReLU()  # 第四个 ReLU 激活函数
        path["conv_4"] = nn.Conv2d(hid_size, out_size, kernel_size=1, padding=0)  # 第四个卷积层（输出层）

        # 使用有序字典定义的层次结构创建一个顺序容器
        self.path = nn.Sequential(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)


class FlavaImageCodebookBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int, **kwargs):
        super().__init__()

        # 计算后增益，用于乘以残差路径的输出
        self.post_gain = 1 / (num_layers**2)

        # 如果输入尺寸不等于输出尺寸，使用 1x1 卷积进行维度匹配
        if in_size != out_size:
            self.id_path = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        else:
            self.id_path = nn.Identity()  # 若输入输出尺寸相同，则使用恒等映射

        # 创建 FLAVA 图像编码块的残差路径
        self.res_path = FlavaImageCodebookResPath(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回恒等映射加上后增益乘以残差路径的输出
        return self.id_path(x) + self.post_gain * self.res_path(x)


class FlavaImageCodebookLayerGroup(nn.Module):
    def __init__(self, num_blocks: int, num_layers: int, in_size: int, out_size: int, use_pool: bool = True):
        super().__init__()
        blocks = OrderedDict()
        
        # 创建多个 FLAVA 图像编码块的组合
        for i in range(num_blocks):
            if i == 0:
                blocks[f"block_{i+1}"] = FlavaImageCodebookBlock(in_size, out_size, num_layers)
            else:
                blocks[f"block_{i+1}"] = FlavaImageCodebookBlock(out_size, out_size, num_layers)

        # 如果指定使用池化层，则添加最大池化层到块组中
        if use_pool:
            blocks["pool"] = nn.MaxPool2d(kernel_size=2)

        # 创建顺序容器，包含所有创建的块组
        self.group = nn.Sequential(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group(x)


# 受 DALL-E 编码器启发，FLAVA 图像码书模型用于生成原始隐藏状态，可用于根据 DALL-E 词汇为图像生成图像标记。用于为 MIM 生成标签。
# 使用 `get_codebook_indices` 函数获取图像的标记。
@add_start_docstrings(
    """
    FLAVA 图像码书模型，受 DALL-E 原始编码器启发而来。输出原始隐藏状态，可用于根据 DALL-E 词汇为图像生成图像标记。用于为 MIM 生成标签。
    使用 `get_codebook_indices` 函数获取图像的标记。
    """,
    FLAVA_START_DOCSTRING.format(config="FlavaImageCodebookConfig"),
)
class FlavaImageCodebook(FlavaPreTrainedModel):
    base_model_prefix = ""
    config_class = FlavaImageCodebookConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(
        self,
        config: FlavaImageCodebookConfig,
        **kwargs: Any,
    ):
        super().__init__(config)  # 调用父类构造函数，初始化模型配置

        self.config = config  # 将配置信息存储到对象属性中
        self.num_groups = config.num_groups  # 设置组数
        self.input_channels = config.input_channels  # 设置输入通道数
        self.num_blocks_per_group = config.num_blocks_per_group  # 设置每组中的块数
        self.hidden_size = config.hidden_size  # 设置隐藏层大小
        self.vocab_size = config.vocab_size  # 设置词汇表大小

        num_layers = self.num_groups * self.num_blocks_per_group  # 计算总层数

        output_blocks = OrderedDict()
        output_blocks["relu"] = nn.ReLU()  # 添加ReLU激活函数到输出块
        output_blocks["conv"] = nn.Conv2d(8 * self.hidden_size, self.vocab_size, kernel_size=1, padding=0)  # 添加卷积层到输出块

        blocks = OrderedDict()
        blocks["input"] = nn.Conv2d(self.input_channels, 1 * self.hidden_size, kernel_size=7, padding=3)  # 添加输入卷积层到块
        blocks["group_1"] = FlavaImageCodebookLayerGroup(
            self.num_blocks_per_group, num_layers, 1 * self.hidden_size, 1 * self.hidden_size
        )  # 添加第一个图像码书层组
        blocks["group_2"] = FlavaImageCodebookLayerGroup(
            self.num_blocks_per_group, num_layers, 1 * self.hidden_size, 2 * self.hidden_size
        )  # 添加第二个图像码书层组
        blocks["group_3"] = FlavaImageCodebookLayerGroup(
            self.num_blocks_per_group, num_layers, 2 * self.hidden_size, 4 * self.hidden_size
        )  # 添加第三个图像码书层组
        blocks["group_4"] = FlavaImageCodebookLayerGroup(
            self.num_blocks_per_group, num_layers, 4 * self.hidden_size, 8 * self.hidden_size, use_pool=False
        )  # 添加第四个图像码书层组，并指定不使用池化操作
        blocks["output"] = nn.Sequential(output_blocks)  # 添加输出块到块序列

        self.blocks = nn.Sequential(blocks)  # 构建模型的块序列

        self.post_init()  # 执行后初始化操作

        if self.config.freeze:
            for param in self.parameters():
                param.requires_grad = False  # 如果配置要求冻结模型，则设置所有参数不需要梯度计算

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
        z_logits = self.blocks(pixel_values)  # 将像素值传递给模型的块序列进行处理，得到logits
        return torch.argmax(z_logits, axis=1)  # 返回logits的最大值索引作为码书索引
        # 使用给定的像素值作为输入，通过神经网络模块生成概率分布的 logits
        z_logits = self.blocks(pixel_values)
        # 对 logits 进行 softmax 处理，得到概率分布
        return nn.Softmax(dim=1)(z_logits)
# 定义一个类，用于处理 Flava 模型的预测头部变换
class FlavaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数，存储在 transform_act_fn 中
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，归一化隐藏状态的维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，对输入的隐藏状态进行变换操作
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 全连接层变换
        hidden_states = self.transform_act_fn(hidden_states)  # 应用激活函数
        hidden_states = self.LayerNorm(hidden_states)  # LayerNorm 归一化
        return hidden_states


# 定义一个类，用于 Flava 模型的蒙版预测头部
class FlavaMaskedPredictionHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        # 创建 FlavaPredictionHeadTransform 实例，用于变换隐藏状态
        self.transform = FlavaPredictionHeadTransform(config)
        # 创建一个线性层，将隐藏状态映射到词汇表大小，无偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个偏置参数
        if weight is not None:
            self.decoder.weight = weight

        # 将 decoder 的偏置参数与 self.bias 关联，以便在 resize_token_embeddings 时正确调整大小
        self.decoder.bias = self.bias

    # 前向传播函数，对输入进行预测头部的操作
    def forward(self, x):
        x = self.transform(x)  # 应用变换操作
        x = self.decoder(x)  # 使用线性层进行预测
        return x


# 定义一个类，用于 Flava 模型的 ITM 头部
class FlavaITMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建 FlavaPooler 实例，用于池化隐藏状态
        self.pooler = FlavaPooler(config)
        # 创建一个线性层，将池化后的隐藏状态映射到 2 个输出类别（用于 ITM 任务）
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播函数，对输入进行 ITM 头部的操作
    def forward(self, x):
        x = self.pooler(x)  # 应用池化操作
        x = self.seq_relationship(x)  # 使用线性层进行序列关系预测
        return x


# 定义一个类，用于 Flava 模型的全局对比头部
class FlavaGlobalContrastiveHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 存储全局背景传播对比的配置
        self.global_backprop_contrastive = config.global_backprop_contrastive
    # 定义一个前向传播函数，接收图片嵌入、文本嵌入和logit缩放因子作为输入参数
    def forward(self, image_embeddings, text_embeddings, logit_scale):
        # 计算温度参数，使用logit缩放因子的指数值
        temperature = torch.exp(logit_scale)
        
        # 检查当前环境是否支持并初始化了分布式训练
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            # 如果未启用分布式训练，则生成标签张量，从0到图片嵌入的数量
            labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
            # 将当前批次的图片嵌入数据存储在列表中
            image_embeddings_all = [image_embeddings]
            # 将当前批次的文本嵌入数据存储在列表中
            text_embeddings_all = [text_embeddings]
        else:
            # 获取当前批次的本地大小
            local_batch_size = image_embeddings.size(0)
            # 获取分布式训练的总节点数
            world_size = torch.distributed.get_world_size()

            if self.global_backprop_contrastive:
                # 如果启用全局反向传播对比，则使用分布式函数收集所有工作节点的图片和文本嵌入
                image_embeddings_all = torch.distributed.nn.functional.all_gather(image_embeddings)
                text_embeddings_all = torch.distributed.nn.functional.all_gather(text_embeddings)
            else:
                # 如果未启用全局反向传播对比，则为每个工作节点创建一个零张量列表
                image_embeddings_all = [torch.zeros_like(text_embeddings) for _ in range(world_size)]
                text_embeddings_all = [torch.zeros_like(image_embeddings) for _ in range(world_size)]
                # 使用分布式函数收集所有工作节点的图片嵌入数据
                torch.distributed.all_gather(image_embeddings_all, image_embeddings)
                # 使用分布式函数收集所有工作节点的文本嵌入数据
                torch.distributed.all_gather(text_embeddings_all, text_embeddings)

            # 为每个本地批次生成对应的标签，考虑当前节点的排名和本地批次大小
            labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
                local_batch_size, device=image_embeddings.device
            )

        # 将收集到的所有图片嵌入数据拼接成一个张量
        image_embeddings_all = torch.cat(image_embeddings_all)
        # 将收集到的所有文本嵌入数据拼接成一个张量
        text_embeddings_all = torch.cat(text_embeddings_all)

        # 计算图片嵌入与所有文本嵌入的点积，并乘以温度参数
        logits_per_image = torch.matmul(image_embeddings, text_embeddings_all.transpose(0, 1)) * temperature
        # 计算文本嵌入与所有图片嵌入的点积，并乘以温度参数
        logits_per_text = torch.matmul(text_embeddings, image_embeddings_all.transpose(0, 1)) * temperature

        # 返回计算得到的图片logits、文本logits以及相应的标签
        return logits_per_image, logits_per_text, labels
# 使用装饰器为 FLAVA 预训练模型添加文档字符串，描述模型输出损失、嵌入、logits 和变换器输出。
@add_start_docstrings(
    """
    The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.
    """,
    FLAVA_START_DOCSTRING.format(config="FlavaConfig") + FLAVA_PRETRAINING_START_DOCSTRING_EXTRA,
)
class FlavaForPreTraining(FlavaPreTrainedModel):
    # 这些键与 xxx.bias 相关联
    _tied_weights_keys = [
        "mmm_text_head.decoder.bias",
        "mmm_image_head.decoder.bias",
        "mlm_head.decoder.bias",
        "mim_head.decoder.bias",
    ]

    def __init__(self, config: FlavaConfig, image_codebook: Optional[nn.Module] = None):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 创建 FLAVA 模型
        self.flava = FlavaModel(config)

        # 设置图像码书，如果未提供且配置指定则初始化图像码书
        self.image_codebook = image_codebook
        if self.image_codebook is None and config.init_codebook:
            self.image_codebook = FlavaImageCodebook(config.image_codebook_config)

        # 根据文本和图像编码器配置创建遮蔽头，以确保有正确的词汇表
        self.mim_head = FlavaMaskedPredictionHead(config.image_config)
        self.mlm_head = FlavaMaskedPredictionHead(config.text_config)
        self.itm_head = FlavaITMHead(config)
        self.mmm_image_head = FlavaMaskedPredictionHead(config.image_config)
        self.mmm_text_head = FlavaMaskedPredictionHead(config.text_config)
        self.global_contrastive_head = FlavaGlobalContrastiveHead(config)

        # 设置图像和文本词汇表大小
        self.image_vocab_size = config.image_config.vocab_size
        self.text_vocab_size = config.text_config.vocab_size
        # 设置 MLM、MIM、全局对比损失权重
        self.mlm_weight = config.mlm_weight
        self.mim_weight = config.mim_weight
        self.global_contrastive_weight = config.global_contrastive_weight
        # 设置交叉熵忽略索引和 ITM 权重
        self.ce_ignore_index = config.ce_ignore_index
        self.itm_weight = config.itm_weight
        # 设置 MMM 图像和文本权重
        self.mmm_image_weight = config.mmm_image_weight
        self.mmm_text_weight = config.mmm_text_weight
        # 设置是否跳过未遮蔽的多模态编码器
        self.skip_unmasked_multimodal_encoder = config.skip_unmasked_multimodal_encoder

        # 执行初始化后操作
        self.post_init()

    def _resize_to_2d(self, x: torch.Tensor):
        # 如果输入张量维度大于 2，则展平为二维张量
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x

    @add_start_docstrings_to_model_forward(
        # 添加模型 forward 方法的输入文档字符串，描述输入参数的形状
        FLAVA_PRETRAINING_INPUTS_DOCSTRING.format("batch_size, text_seq_len", "batch_size, image_num_patches")
    )
    @replace_return_docstrings(output_type=FlavaForPreTrainingOutput, config_class=FlavaConfig)
        # 定义模型的前向传播函数，接收多个输入参数，所有参数都是可选的
        self,
        # 输入的token IDs序列，用于文本输入
        input_ids: Optional[torch.LongTensor] = None,
        # 掩码后的输入token IDs序列，用于MLM任务
        input_ids_masked: Optional[torch.LongTensor] = None,
        # 图像的像素值，用于图像输入
        pixel_values: Optional[torch.FloatTensor] = None,
        # 用于编码图像的码本像素值
        codebook_pixel_values: Optional[torch.FloatTensor] = None,
        # 注意力掩码，用于指示哪些token是padding的
        attention_mask: Optional[torch.Tensor] = None,
        # token类型IDs，用于BERT类型模型
        token_type_ids: Optional[torch.Tensor] = None,
        # 布尔掩码，指示哪些位置是被掩盖的
        bool_masked_pos: Optional[torch.Tensor] = None,
        # 位置IDs，用于指定token的位置信息
        position_ids: Optional[torch.LongTensor] = None,
        # 图像注意力掩码，指示图像中哪些部分需要注意力
        image_attention_mask: Optional[torch.Tensor] = None,
        # 是否跳过未掩盖的多模态编码器
        skip_unmasked_multimodal_encoder: bool = None,
        # MLM任务的标签
        mlm_labels: Optional[torch.Tensor] = None,
        # MIM任务的标签
        mim_labels: Optional[torch.Tensor] = None,
        # ITM任务的标签
        itm_labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: bool = True,
        # 是否返回字典形式的输出
        return_dict: Optional[bool] = None,
        # 是否返回损失值
        return_loss: Optional[bool] = None,
```