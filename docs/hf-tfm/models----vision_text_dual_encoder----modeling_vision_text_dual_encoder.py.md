# `.\transformers\models\vision_text_dual_encoder\modeling_vision_text_dual_encoder.py`

```
# 设置文件编码为 utf-8
# 版权声明
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的适用性和
# 适销性。请查看许可证以获取特定语言的权限和限制。
""" PyTorch VisionTextDualEncoder model."""

# 导入必要的库
from typing import Optional, Tuple, Union
import torch
from torch import nn
# 导入自定义的模块
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from ..clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "VisionTextDualEncoderConfig"

# VisionTextDualEncoder 类的起始文档字符串
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~AutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.

    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.

    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
```  
    # 参数说明：
    # config ([`VisionEncoderDecoderConfig`]): 包含模型所有参数的模型配置类。
    # 使用配置文件初始化不会加载与模型相关的权重，只会加载配置信息。
    # 查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 定义了用于双编码器模型的文本输入参数的文档字符串
VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：

            - 对于**未屏蔽**的标记为 1，
            - 对于**屏蔽**的标记为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义了用于双编码器模型的视觉输入参数的文档字符串
VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，将忽略填充。像素值可以使用 [`AutoImageProcessor`] 获取。有关详细信息，请参阅 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义了用于双编码器模型的输入参数的文档字符串
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下将忽略填充。可以使用图像处理器获取像素值（例如，如果您使用 ViT 作为编码器，则应使用 [`AutoImageProcessor`]）。有关详细信息，请参阅 [`ViTImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 从transformers.models.clip.modeling_clip.contrastive_loss中复制的对比损失函数
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 使用交叉熵损失计算对比损失
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 从transformers.models.clip.modeling_clip.clip_loss中复制的CLIP损失函数
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算文本损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像损失
    image_loss = contrastive_loss(similarity.t())
    # 返回文本损失和图像损失的平均值
    return (caption_loss + image_loss) / 2.0


# 根据VISION_TEXT_DUAL_ENCODER_START_DOCSTRING添加文档字符串的类
class VisionTextDualEncoderModel(PreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = "vision_text_dual_encoder"

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        # 如果没有提供配置或视觉和文本模型，则引发错误
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            # 如果没有提供配置，则根据视觉和文本模型的配置创建配置
            config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # 使用配置初始化
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        # 确保各个模型的配置引用共享配置，以便配置的更新会同步
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        # 定义视觉和文本的投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    # 添加文档字符串到模型前向传播函数
    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer

        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VisionTextDualEncoderModel, AutoImageProcessor

        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入的像素值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        return_loss: Optional[bool] = None,  # 是否返回损失
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 目前不支持快速初始化用于复合模型
        kwargs["_fast_init"] = False
        # 调用父类的方法来加载预训练模型
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,  # 视觉模型的名称或路径
        text_model_name_or_path: str = None,  # 文本模型的名称或路径
        *model_args,  # 模型参数
        **kwargs,  # 其他关键字参数
```