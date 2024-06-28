# `.\models\vision_text_dual_encoder\modeling_vision_text_dual_encoder.py`

```py
# 设置编码格式为 UTF-8

# 版权声明和许可协议，表明此代码的版权和许可情况
# 版权所有 2021 年 HuggingFace Inc. 团队。保留所有权利。
# 根据 Apache 许可证 2.0 版本授权，除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，
# 不提供任何明示或暗示的保证或条件。有关详细信息，请参阅许可证。

""" PyTorch VisionTextDualEncoder model. """

# 导入必要的库
from typing import Optional, Tuple, Union

import torch
from torch import nn

# 导入模型相关的实用函数和类
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

# 自动导入相关配置类
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel

# 导入与 CLIP 相关的模型和配置
from ..clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel

# 导入 VisionTextDualEncoder 模型的配置类
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档用变量，指定 VisionTextDualEncoderConfig 的字符串
_CONFIG_FOR_DOC = "VisionTextDualEncoderConfig"

# VisionTextDualEncoder 类的文档字符串，提供了关于该类的详细信息和用法示例
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
    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
Args:
    input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        输入序列标记在词汇表中的索引。默认情况下，将忽略填充标记。

        可以使用[`PreTrainedTokenizer`]获取索引。参见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]了解详情。

        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        遮盖机制，用于避免在填充标记索引上执行注意力操作。遮盖值选择在 `[0, 1]`：

        - 1 表示**未遮盖**的标记，
        - 0 表示**遮盖**的标记。

        [什么是注意力遮盖？](../glossary#attention-mask)
    position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        输入序列中每个标记的位置索引，用于位置嵌入。选择范围在 `[0, config.max_position_embeddings - 1]`。

        [什么是位置 ID？](../glossary#position-ids)
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 获取更多细节。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 获取更多细节。
    return_dict (`bool`, *optional*):
        是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记的索引，用于表示词汇表中的每个标记。默认情况下会忽略填充。
            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取详情。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充的标记索引上执行注意力操作的掩码。掩码的值选择在 `[0, 1]` 之间：
            # - 对于 **未屏蔽的** 标记，设为 1，
            # - 对于 **屏蔽的** 标记，设为 0。
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]` 之间。
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充。可以通过图像处理器获取像素值（例如，如果使用 ViT 作为编码器，应使用 `AutoImageProcessor`）。
            # 详见 `ViTImageProcessor.__call__` 获取详情。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 字段提供更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 字段提供更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通元组。
"""
Copied from transformers.models.clip.modeling_clip.contrastive_loss
定义对比损失函数，输入为 logits，输出为损失值
"""
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


"""
Copied from transformers.models.clip.modeling_clip.clip_loss
定义 CLIP 损失函数，输入为相似性张量，输出为损失值
"""
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算文本和图像的对比损失
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    # 返回文本和图像损失的平均值
    return (caption_loss + image_loss) / 2.0


"""
@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
双编码器模型，结合视觉和文本输入进行编码
"""
class VisionTextDualEncoderModel(PreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = "vision_text_dual_encoder"

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            # 如果未提供配置，则从视觉和文本模型的配置创建配置对象
            config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # 使用父类初始化模型
        super().__init__(config)

        # 如果未提供视觉模型，则根据配置创建默认的视觉模型
        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        # 如果未提供文本模型，则根据配置创建默认的文本模型
        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        # 将创建的视觉模型和文本模型保存到当前对象中
        self.vision_model = vision_model
        self.text_model = text_model

        # 确保各个模型的配置对象与共享的配置对象同步更新
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        # 设置视觉和文本嵌入的维度和投影维度
        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        # 定义视觉和文本的线性投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 初始化 logits 缩放参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # 注意：这里函数定义没有完全列出，后续可能还有参数
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    # 使用装饰器添加模型前向传播的文档字符串，文档字符串定义了输入参数和返回结果的形状和含义
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

        ```
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
        # 使用视觉模型处理像素值，返回视觉特征，可以控制是否输出注意力和隐藏状态，并选择返回形式
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉输出的第二个元素中获取池化输出作为特征表示
        pooled_output = vision_outputs[1]  # pooled_output
        # 将池化输出应用于视觉投影层，得到最终的图像特征表示
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征表示
        return image_features
    # 定义一个类方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为长整型张量，可选
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入的像素值，类型为浮点张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，类型为张量，可选
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为长整型张量，可选
        return_loss: Optional[bool] = None,  # 是否返回损失值，类型为布尔值，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # Token 类型 IDs，类型为长整型张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为布尔值，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，类型为布尔值，可选
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 目前不支持复合模型的快速初始化
        kwargs["_fast_init"] = False
        # 调用父类的 `from_pretrained` 方法，并传递所有的位置参数和关键字参数
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,  # 视觉模型的名称或路径，类型为字符串，可选
        text_model_name_or_path: str = None,  # 文本模型的名称或路径，类型为字符串，可选
        *model_args,  # 其他模型参数，位置参数的元组
        **kwargs,  # 其他模型参数，关键字参数的字典
```