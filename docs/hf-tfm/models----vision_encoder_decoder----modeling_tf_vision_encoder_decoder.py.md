# `.\transformers\models\vision_encoder_decoder\modeling_tf_vision_encoder_decoder.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache许可证2.0版授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
""" 用于支持TF Vision-Encoder-Text-Decoder架构的类"""


from __future__ import annotations

import re
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...configuration_utils import PretrainedConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFPreTrainedModel, get_initializer, unpack_inputs
from ...tf_utils import shape_list
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置
_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

# 弃用警告
DEPRECATION_WARNING = (
    "Version v4.17.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.17.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

# 文档起始字符串
VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~TFAutoModel.from_pretrained`] function and the decoder is loaded via [`~TFAutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    # 在这篇论文中展示了如何利用大型预训练视觉模型进行光学字符识别（OCR），从而显著提高性能。
    
    # 在训练/微调了这样一个视觉-编码器-文本-解码器模型之后，它可以像其他模型一样保存/加载（查看示例以获取更多信息）。
    
    # 这个模型继承自[`TFPreTrainedModel`]。查看超类文档以了解库为所有模型实现的通用方法（如下载或保存、调整输入嵌入、修剪头等）。
    
    # 这个模型也是一个[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。将其用作常规的TF 2.0 Keras模型，并参考TF 2.0文档以获取与一般用法和行为相关的所有信息。
    
    # 参数:
    #     config ([`VisionEncoderDecoderConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看[`~TFPreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

# 定义了一个文档字符串常量，用于描述视觉编码器-解码器模型的输入
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

# 从transformers.models.encoder_decoder.modeling_tf_encoder_decoder.shift_tokens_right中复制函数
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 如果pad_token_id为None，则引发值错误
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)

    # 如果decoder_start_token_id为None，则引发值错误
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)

    # 创建起始标记
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    # 将输入ID向右移动一个位置
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将标签中可能存在的-100值替换为pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # 断言`labels`中只有正值和-100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保通过包装结果在一个身份无操作中调用断言操作
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids

# 添加起始文档字符串
@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class TFVisionEncoderDecoderModel(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    r"""
    [`TFVisionEncoderDecoderModel`]是一个通用的模型类，当使用[`~TFAutoModel.from_pretrained`]类方法为编码器创建一个库中的基本视觉模型类，并为解码器创建另一个基本模型类时，将实例化为一个transformer架构。
    """

    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    load_weight_prefix = "tf_vision_encoder_decoder_model"
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[TFPreTrainedModel] = None,
        decoder: Optional[TFPreTrainedModel] = None,
        # 如果没有提供配置信息并且编码器或解码器为空，则抛出数值错误
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        # 如果没有提供配置信息，则从编码器和解码器的配置中创建一个视觉编码器解码器配置
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            # 如果配置不是指定的配置类，则抛出数值错误
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # 如果解码器的交叉注意力隐藏大小不为空
        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器的交叉注意力隐藏大小不等于编码器的隐藏大小，则抛出数值错误
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 使用配置初始化
        super().__init__(config)

        # 如果编码器为空，则从配置中创建一个自动模型编码器
        if encoder is None:
            encoder = TFAutoModel.from_config(config.encoder, name="encoder")

        # 如果解码器为空，则从配置中创建一个自动模型用于因果语言建模
        if decoder is None:
            decoder = TFAutoModelForCausalLM.from_config(config.decoder, name="decoder")

        # 设置编码器和解码器
        self.encoder = encoder
        self.decoder = decoder

        # 如果编码器的配置与共享的编码器配置不同，则发出警告
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        # 如果解码器的配置与共享的解码器配置不同，则发出警告
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # 确保各个模型的配置引用共享配置，以便配置的更新会同步
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # 如果编码器输出可能需要投影到不同维度以供解码器使用
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = tf.keras.layers.Dense(
                units=self.decoder.config.hidden_size,
                kernel_initializer=get_initializer(config.encoder.initializer_range),
                name="enc_to_dec_proj",
            )

        # 如果编码器有输出嵌入，则抛出数值错误
        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )
    # 定义一个方法，返回输入的签名
    def input_signature(self):
        # 获取视觉编码器的配置
        vision_config = self.config.encoder
        # 如果视觉配置中有"vision_config"属性，则将其赋值给vision_config
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config
        # 如果视觉配置中有"image_size"属性，则将其赋值给image_size，否则使用input_size
        if hasattr(vision_config, "image_size"):
            image_size = vision_config.image_size
        else:
            image_size = vision_config.input_size
        # 返回包含像素值和解码器输入ID的字典
        return {
            "pixel_values": tf.TensorSpec(
                shape=(
                    None,
                    vision_config.num_channels,
                    image_size,
                    image_size,
                ),
                dtype=tf.float32,
            ),
            "decoder_input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="decoder_input_ids"),
        }

    # 返回编码器
    def get_encoder(self):
        return self.encoder

    # 返回解码器
    def get_decoder(self):
        return self.decoder

    # 返回输入嵌入
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    # 返回输出嵌入
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 将TF权重重命名为PT权重
    def tf_to_pt_weight_rename(self, tf_weight):
        # 重命名TF和PT权重以匹配
        encoder_model_type = self.config.encoder.model_type
        if "encoder" in tf_weight and "decoder" not in tf_weight:
            return (re.sub(rf"encoder\.{encoder_model_type}\.", "encoder.", tf_weight),)
        else:
            return (tf_weight,)

    # 从预训练的编码器和解码器创建模型
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型进行推理
    def call(
        self,
        pixel_values: np.ndarray | tf.Tensor | None = None,  # 输入像素值
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 解码器输入的标识符
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的注意力掩码
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,  # 编码器的输出
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 解码器的输入嵌入
        labels: np.ndarray | tf.Tensor | None = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        training: bool = False,  # 是否处于训练模式
        **kwargs,  # 其他参数
    # 定义一个方法，用于生成模型的输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则获取过去的键值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.decoder.use_cache else None
        # 如果配置中输出隐藏状态，则转换为张量
        dec_hs = (
            tf.convert_to_tensor(output.decoder_hidden_states) if self.config.decoder.output_hidden_states else None
        )
        # 如果配置中输出解码器注意力，则转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.decoder.output_attentions else None
        # 如果配置中输出编码器隐藏状态，则转换为张量
        enc_hs = (
            tf.convert_to_tensor(output.encoder_hidden_states) if self.config.encoder.output_hidden_states else None
        )
        # 如果配置中输出编码器注意力，则转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.encoder.output_attentions else None
        # 如果配置中输出交叉注意力，并且交叉注意力不为空，则转换为张量
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.decoder.output_attentions and output.cross_attentions is not None
            else None
        )

        # 返回序列到序列模型的输出
        return TFSeq2SeqLMOutput(
            logits=output.logits,  # 输出的逻辑值
            past_key_values=pkv,  # 过去的键值
            decoder_hidden_states=dec_hs,  # 解码器隐藏状态
            decoder_attentions=dec_attns,  # 解码器注意力
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器最后的隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器隐藏状态
            encoder_attentions=enc_attns,  # 编码器注意力
            cross_attentions=cross_attns,  # 交叉注意力
        )

    # 定义一个方法，用于为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
        ):
        # 准备解码器输入，为生成做准备
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 获取解码器的注意力掩码
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 获取过去的键值
        past_key_values = decoder_inputs.get("past_key_values")
        # 构建输入字典
        input_dict = {
            "pixel_values": None,  # 需要传递以使 Keras.layer.__call__ 正常运行
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            # TODO (joao): 在生成重构完成后，`TFBaseModelOutput` 包装器应该不再需要
            "encoder_outputs": TFBaseModelOutput(last_hidden_state=encoder_outputs[0]),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        return input_dict

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 从标签中准备解码器输入的 ID
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误，不支持通过 TFVisionEncoderDecoderModel 直接调整嵌入层大小
        raise NotImplementedError(
            "Resizing the embedding layers via the TFVisionEncoderDecoderModel directly is not supported. "
            "Please use the respective methods of the wrapped objects (model.decoder.resize_token_embeddings(...))"
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 enc_to_dec_proj 属性，则构建它
        if getattr(self, "enc_to_dec_proj", None) is not None:
            with tf.name_scope(self.enc_to_dec_proj.name):
                self.enc_to_dec_proj.build([None, None, self.encoder.config.hidden_size])
        # 如果存在 encoder 属性，则构建它
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 decoder 属性，则构建它
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
```