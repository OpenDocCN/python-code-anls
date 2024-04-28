# `.\transformers\models\vision_text_dual_encoder\modeling_tf_vision_text_dual_encoder.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""TensorFlow VisionTextDualEncoder model."""

# 导入必要的库
from __future__ import annotations
import re
from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 导入 Hugging Face 库中的相关模块和函数
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_utils import TFPreTrainedModel, unpack_inputs
from ...tf_utils import shape_list
from ...utils import (
    DUMMY_INPUTS,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel
from ..clip.modeling_tf_clip import CLIPVisionConfig, TFCLIPOutput, TFCLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "VisionTextDualEncoderConfig"

# VisionTextDualEncoder 类的起始文档字符串
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~TFAutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.

    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.

    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a
    regular Keras Model and refer to the TF documentation for all matter related to general usage and behavior.
``` 
    # 参数说明：
    # config ([`VisionEncoderDecoderConfig`]): 模型配置类，包含模型的所有参数。
    # 使用配置文件初始化不会加载与模型相关的权重，只会加载配置信息。
    # 查看 [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 定义用于文本输入的参数说明文档字符串
VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            文本输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：

            - 对于**未屏蔽**的标记为 1，
            - 对于**屏蔽**的标记为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义用于视觉输入的参数说明文档字符串
VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，将忽略填充。像素值可以使用 [`AutoImageProcessor`] 获取。有关详细信息，请参阅 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义用于双编码器输入的参数说明文档字符串
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下，将忽略填充。可以使用图像处理器获取像素值（例如，如果您使用 ViT 作为编码器，则应使用 [`AutoImageProcessor`]）。有关详细信息，请参阅 [`ViTImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 定义对比损失函数，输入为 logits，输出为损失值
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    # 计算稀疏分类交叉熵损失的平均值
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )

# 定义 CLIP 损失函数，输入为相似度，输出为损失值
def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    # 计算描述文本的对比损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像的对比损失
    image_loss = contrastive_loss(tf.transpose(similarity))
    # 返回描述文本和图像对比损失的平均值
    return (caption_loss + image_loss) / 2.0

# 定义 TFVisionTextDualEncoderModel 类，继承自 TFPreTrainedModel
@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class TFVisionTextDualEncoderModel(TFPreTrainedModel):
    # 配置类为 VisionTextDualEncoderConfig
    config_class = VisionTextDualEncoderConfig
    # 基础模型前缀为 vision_text_dual_encoder
    base_model_prefix = "vision_text_dual_encoder"
    # 加载权重前缀为 tf_vision_text_dual_encoder_model

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[TFPreTrainedModel] = None,
        text_model: Optional[TFPreTrainedModel] = None,
    ):
        # 如果未提供配置且未提供视觉模型和文本模型，则引发错误
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        # 如果未提供配置，则根据视觉模型和文本模型的配置创建配置
        if config is None:
            config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            # 如果配置不是 VisionTextDualEncoderConfig 类型，则引发错误
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # 使用配置初始化
        super().__init__(config)

        # 如果未提供视觉模型，则根据配置创建视觉模型
        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = TFCLIPVisionModel.from_config(config.vision_config, name="vision_model")
            else:
                vision_model = TFAutoModel.from_config(config.vision_config, name="vision_model")

        # 如果未提供文本模型，则根据配置创建文本模型
        if text_model is None:
            text_model = TFAutoModel.from_config(config.text_config, name="text_model")

        # 设置视觉模型和文本模型
        self.vision_model = vision_model
        self.text_model = text_model

        # 确保各个模型的配置引用共享配置，以便配置更新同步
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        # 设置视觉嵌入维度、文本嵌入维度和投影维度
        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        # 创建视觉投影层和文本投影层
        self.visual_projection = Dense(self.projection_dim, use_bias=False, name="visual_projection")
        self.text_projection = Dense(self.projection_dim, use_bias=False, name="text_projection")
        self.logit_scale = None
        self.config = config
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 使用 Constant 初始化器创建 logit_scale 权重
        initializer = tf.keras.initializers.Constant(self.config.logit_scale_init_value)
        self.logit_scale = self.add_weight(shape=(1,), initializer=initializer, name="logit_scale")

        # 如果存在 visual_projection 属性，则构建它
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        # 如果存在 text_projection 属性，则构建它
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])
        # 使用 tf.name_scope 构建 vision_model
        with tf.name_scope(self.vision_model.name):
            self.vision_model.build(None)
        # 使用 tf.name_scope 构建 text_model
        with tf.name_scope(self.text_model.name):
            self.text_model.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        # 重命名 TF 权重以匹配 PT 权重
        # TF 和 PT 权重不对齐是因为我们的 TF 基类比 PT 模型多了一层
        # 如果去掉这一层，权重名称就会正常对齐
        if "vision_model" in tf_weight:
            if tf_weight.count("vision_model") == 1:
                return (re.sub(r"vision_model\..*?\.", "vision_model.", tf_weight),)
            elif tf_weight.count("vision_model") == 2:
                return (re.sub(r"vision_model\..*?\.vision_model", "vision_model.vision_model", tf_weight),)
            else:
                raise ValueError(
                    f"Unexpected weight name {tf_weight}. Please file an issue on the"
                    " Transformers repo to let us know about this error!"
                )
        elif "text_model" in tf_weight:
            return (re.sub(r"text_model\..*?\.", "text_model.", tf_weight),)
        else:
            return (tf_weight,)

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
    ):
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPTextModel`].

        Examples:

        ```py
        >>> from transformers import TFVisionTextDualEncoderModel, AutoTokenizer

        >>> model = TFVisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", from_pt=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="np")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 获取文本特征，通过将投影层应用于[`TFCLIPTextModel`]的汇总输出获得
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取汇总输出
        pooled_output = text_outputs[1]
        # 将汇总输出应用于文本投影层，得到文本特征
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
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPVisionModel`].

        Examples:

        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import TFVisionTextDualEncoderModel, AutoImageProcessor

        >>> model = TFVisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", from_pt=True)
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="np")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # 获取图像特征，通过将投影层应用于[`TFCLIPVisionModel`]的汇总输出获得
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉输出中获取汇总输出
        pooled_output = vision_outputs[1]  # pooled_output
        # 将汇总输出应用于视觉投影层，得到图像特征
        image_features = self.visual_projection(pooled_output)

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: tf.Tensor | None = None,  # 输入的 token IDs 张量
        pixel_values: tf.Tensor | None = None,  # 输入的像素值张量
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量
        position_ids: tf.Tensor | None = None,  # 位置 IDs 张量
        return_loss: Optional[bool] = None,  # 是否返回损失值
        token_type_ids: tf.Tensor | None = None,  # token 类型 IDs 张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        training: bool = False,  # 是否处于训练模式
    # 定义一个类方法，用于从预训练的视觉文本模型中加载模型
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,  # 视觉模型的名称或路径
        text_model_name_or_path: str = None,  # 文本模型的名称或路径
        *model_args,  # 其他模型参数
        **kwargs,  # 其他关键字参数
    # 定义一个属性，用于生成网络的虚拟输入
    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        # 创建一个虚拟的 token IDs 张量
        input_ids = tf.constant(DUMMY_INPUTS, dtype=tf.int32)
        batch_size, seq_len = input_ids.shape

        # 创建一个虚拟的像素值张量
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(
                batch_size,
                self.config.vision_config.num_channels,
                self.config.vision_config.image_size,
                self.config.vision_config.image_size,
            ),
            dtype=tf.float32,
        )
        pixel_values = tf.constant(VISION_DUMMY_INPUTS)
        # 组合虚拟输入数据
        dummy = {"pixel_values": pixel_values, "input_ids": input_ids}
        return dummy
```