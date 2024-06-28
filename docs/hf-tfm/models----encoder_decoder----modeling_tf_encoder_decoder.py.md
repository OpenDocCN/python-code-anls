# `.\models\encoder_decoder\modeling_tf_encoder_decoder.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有，采用 Apache License, Version 2.0
# 只有在遵循许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则不得使用本文件中的代码
# 本文件中的代码按"原样"提供，不提供任何形式的担保或条件，无论是明示的还是暗示的
# 详细信息请参阅许可证
""" Classes to support TF Encoder-Decoder architectures"""

from __future__ import annotations  # 支持在注解中使用自身类名

import inspect  # 导入用于获取对象信息的模块
import re  # 导入正则表达式模块
import warnings  # 导入警告处理模块
from typing import Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 数学库
import tensorflow as tf  # 导入 TensorFlow 深度学习库

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput  # 导入 TensorFlow 模型输出类
from ...modeling_tf_utils import (  # 导入 TensorFlow 模型工具函数
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    unpack_inputs,
)
from ...tf_utils import shape_list  # 导入获取张量形状的函数
from ...utils import (  # 导入实用函数
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig  # 导入自动配置类
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM  # 导入自动 TensorFlow 模型类
from .configuration_encoder_decoder import EncoderDecoderConfig  # 导入编码解码器配置类

logger = logging.get_logger(__name__)  # 获取日志记录器对象

_CONFIG_FOR_DOC = "EncoderDecoderConfig"  # 用于文档的配置名称

DEPRECATION_WARNING = (
    "Version v4.17.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.17.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)  # 弃用警告信息

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~TFAutoModel.from_pretrained`] function and the decoder is loaded via [`~TFAutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models

```  
    (see the examples for more information).



    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)



    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.



    Parameters:
        config ([`EncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

# 定义一个函数，用于将输入的token_ids向右移动，模拟decoder端的输入
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 如果pad_token_id未设置，则抛出数值错误
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将pad_token_id转换为与input_ids相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)

    # 如果decoder_start_token_id未设置，则抛出数值错误
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 将decoder_start_token_id转换为与input_ids相同的数据类型
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)

    # 创建一个形状为(input_ids的行数, 1)的张量，每个元素均为decoder_start_token_id
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    # 将start_tokens与input_ids的前几列合并，构成向右移动后的输入token_ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将shifted_input_ids中可能的-100值替换为pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # 断言shifted_input_ids中的值均大于等于0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作的调用，通过在结果中包装一个身份无操作
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
# TFEncoderDecoderModel类，继承自TFPreTrainedModel和TFCausalLanguageModelingLoss
class TFEncoderDecoderModel(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    r"""
    [`TFEncoderDecoderModel`]是一个通用的模型类，创建时会使用库中的一个基础模型类作为encoder和另一个作为decoder，
    使用[`~TFAutoModel.from_pretrained`]类方法创建encoder，和使用[`~TFAutoModelForCausalLM.from_pretrained`]类方法创建decoder。
    """

    # 类属性，指定配置类为EncoderDecoderConfig
    config_class = EncoderDecoderConfig
    # 模型前缀为"encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # 加载权重时的前缀为"tf_encoder_decoder_model"
    load_weight_prefix = "tf_encoder_decoder_model"

    # 初始化方法
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[TFPreTrainedModel] = None,
        decoder: Optional[TFPreTrainedModel] = None,
    ):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    # 获取encoder方法
    def get_encoder(self):
        return self.encoder

    # 获取decoder方法
    def get_decoder(self):
        return self.decoder

    # 获取输入嵌入方法，委托给encoder的get_input_embeddings方法
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    # 获取输出嵌入方法，委托给decoder的get_output_embeddings方法
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入方法，委托给decoder的set_output_embeddings方法
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)
    # Matt: The TF and PT weights don't align because our TF base classes have an extra layer compared to PT models
    # (the main model stem is in the MainLayer class). If we remove that layer, then weight names sync up as normal.
    # However, the name of that extra layer is the name of the MainLayer in the base model. We make the assumption
    # here that the config model_type is the same as the name of the MainLayer. I don't know of anywhere that's
    # not the case, and I wasn't sure how else to go from the config to the correct MainLayer name!
    def tf_to_pt_weight_rename(self, tf_weight):
        # 函数用于重命名 TF 到 PT 权重的函数。由于 TF 基类比 PT 模型多了一个层（主模型干部在 MainLayer 类中），
        # 导致权重名称不匹配。如果去除这一层，则权重名称会正常对应。假设配置文件中的 model_type 和 MainLayer 名称相同，
        # 因此在此处进行这一假设。不清楚是否存在不符合这一假设的情况，也不确定如何从配置中得到正确的 MainLayer 名称。

        # This override is only needed in the case where we're crossloading weights from PT. However, since weights are
        # often safetensors now, we don't know if we're going to be crossloading until we sniff the weights file.
        # Therefore, we specify tf_to_pt_weight_rename anyway, and let the super method figure out if it needs it
        # or not.
        # 此重写仅在从 PT 加载权重时需要。但是，由于现在权重通常是 SafeTensor，因此在嗅探权重文件之前，我们不知道是否会进行跨加载。
        # 因此，我们仍然指定 tf_to_pt_weight_rename，让超类方法决定是否需要使用它。
        encoder_model_type = self.config.encoder.model_type
        if "encoder" in tf_weight and "decoder" not in tf_weight:
            # 如果权重名称中包含 "encoder" 但不包含 "decoder"，则替换名称中的 encoder.model_type 部分为 encoder.
            return (re.sub(rf"encoder\.{encoder_model_type}\.", "encoder.", tf_weight),)
        else:
            # 否则直接返回原始的 TF 权重名称
            return (tf_weight,)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    # 函数装饰器，用于从预训练的编码器-解码器模型名称或路径创建模型的类方法。
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    # 函数装饰器，添加到 call 方法，用于指定输入、输出和返回文档字符串的生成和替换规则。
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    # 函数用于为生成准备输入，接受输入 ID、过去的关键值、注意力掩码、缓存标志、编码器输出等参数。
    ):
        # 准备解码器生成所需的输入
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 获取解码器的注意力遮罩，如果存在的话
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 获取过去的关键值
        past_key_values = decoder_inputs.get("past_key_values")
        # 如果过去的关键值不存在，则在TF GPT2上获取过去的值
        if past_key_values is None:
            past_key_values = decoder_inputs.get("past")  # 例如在TF GPT2上
        # 构建输入字典，准备传递给Keras.layer.__call__以确保正常工作
        input_dict = {
            "input_ids": None,  # 需要传递以使Keras.layer.__call__工作正常
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            # TODO (joao): 在生成重构完成后，不应再需要`TFBaseModelOutput`包装器
            "encoder_outputs": TFBaseModelOutput(last_hidden_state=encoder_outputs[0]),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        return input_dict

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 根据标签为解码器的输入准备输入ID
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误，直接调整嵌入层不支持通过TFEncoderDecoderModel
        raise NotImplementedError(
            "Resizing the embedding layers via the TFEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # 在此应用解码器缓存重新排序
        return self.decoder._reorder_cache(past, beam_idx)

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在enc_to_dec_proj属性，则构建编码器到解码器的投影
        if getattr(self, "enc_to_dec_proj", None) is not None:
            with tf.name_scope(self.enc_to_dec_proj.name):
                self.enc_to_dec_proj.build([None, None, self.encoder.config.hidden_size])
        # 如果存在encoder属性，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在decoder属性，则构建解码器
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
```