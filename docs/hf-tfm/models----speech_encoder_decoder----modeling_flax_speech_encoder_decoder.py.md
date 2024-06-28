# `.\models\speech_encoder_decoder\modeling_flax_speech_encoder_decoder.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Flax Speech-Encoder-Decoder architectures"""

import os
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig


logger = logging.get_logger(__name__)

# 用于文档字符串的配置名称
_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"

# 文档字符串的起始部分，描述了 Speech-Encoder-Decoder 模型的初始化和用途
SPEECH_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a speech-sequence-to-text-sequence model with any pretrained speech
    autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is
    loaded via [`~AutoModel.from_pretrained`] function and the decoder is loaded via
    [`~AutoModelForCausalLM.from_pretrained`] function. Cross-attention layers are automatically added to the decoder
    and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [Large-Scale Self- and Semi-Supervised Learning for Speech
    Translation](https://arxiv.org/abs/2104.06678) it is shown how leveraging large pretrained speech models for speech
    translation yields a significant performance improvement.

    After such an Speech-Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).
    # 这个模型继承自 `FlaxPreTrainedModel`。请查看超类文档以了解库为所有模型实现的通用方法（如下载或保存模型、调整输入嵌入、修剪头等）。
    
    # 这个模型同时也是 Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 的子类。
    # 可以像常规的 Flax Module 一样使用它，并参考 Flax 文档了解与一般用法和行为相关的所有事项。

    # 参数:
    # config ([`SpeechEncoderDecoderConfig`]): 模型配置类，包含模型的所有参数。
    # 初始化时使用配置文件不会加载与模型关联的权重，只加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    # 计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）之一。
    
    # 这可以用于在GPU或TPU上启用混合精度训练或半精度推理。如果指定了dtype，所有计算将使用给定的 `dtype` 执行。
    
    # **请注意，这只指定计算的数据类型，并不影响模型参数的数据类型。**
    
    # 如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

SPEECH_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        inputs (`jnp.ndarray` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*):
            Float values of input raw speech waveform or speech features. Values can be obtained by loading a `.flac`
            or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile
            library (`pip install soundfile`). To prepare the array into `inputs`, either the [`Wav2Vec2Processor`] or
            [`Speech2TextProcessor`] should be used for padding and conversion into a tensor of type
            `torch.FloatTensor`.
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For sequence to sequence training, `decoder_input_ids` should be provided. `decoder_input_ids` should be
            created outside of the model by shifting the `labels` to the right, replacing -100 by the `pad_token_id`
            and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.decoder.max_position_embeddings - 1]`.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.
"""
    # 定义函数签名，描述了函数的输入参数及其类型
    Args:
        inputs (`jnp.ndarray` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*):
            Float values of input raw speech waveform or speech features. Values can be obtained by loading a *.flac*
            or *.wav* audio file into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile
            library (*pip install soundfile*). To prepare the array into *inputs*, either the [`Wav2Vec2Processor`] or
            [`Speech2TextProcessor`] should be used for padding and conversion into a tensor of type
            *torch.FloatTensor*.
    
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
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
            If set to `True`, the model will return a [`~utils.FlaxBaseModelOutput`] instead of a plain tuple.
"""
定义了一个多行字符串常量，用作文档字符串，描述了输入解码器的期望格式。
"""


class FlaxSpeechEncoderDecoderModule(nn.Module):
    config: SpeechEncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder

        # 从`modeling_hybrid_clip.py`中复制代码，并进行修改。
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING

        # 根据编码器和解码器配置，选择相应的模块类
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        # 使用选择的模块类和配置创建编码器和解码器实例
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # 如果编码器输出维度与解码器隐藏状态维度不同，并且解码器没有交叉注意力隐藏状态
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # 创建一个全连接层，用于将编码器输出投影到解码器期望的维度
            self.enc_to_dec_proj = nn.Dense(
                self.decoder.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range),
                dtype=self.dtype,
            )
        else:
            self.enc_to_dec_proj = None

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        计算卷积层的输出长度
        """

        add_adapter = self.config.encoder.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的一维卷积层输出长度公式
            return (input_length - kernel_size) // stride + 1

        # 遍历编码器的卷积核和步长，计算每个卷积层的输出长度
        for kernel_size, stride in zip(self.config.encoder.conv_kernel, self.config.encoder.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要，添加适配器卷积层，并计算适配器卷积层的输出长度
        if add_adapter:
            for _ in range(self.config.encoder.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.encoder.adapter_stride)

        return input_lengths

    def _get_encoder_module(self):
        return self.encoder

    def _get_projection_module(self):
        return self.enc_to_dec_proj

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        inputs,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        encoder_outputs=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        freeze_feature_encoder: bool = False,
        ):
            # 如果 encoder_outputs 为 None，则调用 self.encoder 进行编码器的前向传播计算
            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
                freeze_feature_encoder=freeze_feature_encoder,
            )

        # 获取编码器的隐藏状态
        encoder_hidden_states = encoder_outputs[0]

        # 如果存在 enc_to_dec_proj，进行编码器隐藏状态的投影
        if self.enc_to_dec_proj is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # 计算正确的编码器注意力掩码
        if attention_mask is not None:
            # 调用 self.encoder 的 _get_feature_vector_attention_mask 方法获取特征向量注意力掩码
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        # 在 flax 脚本 modeling_flax_wav2vec2.py 中执行解码器的前向传播计算
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果 return_dict 为 False，则返回解码器和编码器的输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回 FlaxSeq2SeqLMOutput 对象，包含解码器的输出和编码器的相关隐藏状态和注意力
        return FlaxSeq2SeqLMOutput(
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 导入FlaxSpeechEncoderDecoderModel类所需的文档字符串
@add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
# 定义FlaxSpeechEncoderDecoderModel类，继承自FlaxPreTrainedModel类
class FlaxSpeechEncoderDecoderModel(FlaxPreTrainedModel):
    # 类的描述性文档字符串，解释此类是如何实例化的，使用了transformer架构作为编码器和解码器模块
    r"""
    [`FlaxSpeechEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture
    with the module (flax.nn.Module) of one of the base model classes of the library as encoder module and another one
    as decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method for the
    encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # 引用此类的配置类，即SpeechEncoderDecoderConfig类
    config_class = SpeechEncoderDecoderConfig
    # 指定模型的基础名称前缀，用于命名模型的不同部分
    base_model_prefix: str = "speech_encoder_decoder"
    # 引用此类使用的模块类，即FlaxSpeechEncoderDecoderModule类
    module_class = FlaxSpeechEncoderDecoderModule

    # 初始化方法，接受多个参数，包括配置、输入形状、随机种子、数据类型等
    def __init__(
        self,
        config: SpeechEncoderDecoderConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果_do_init为False，则抛出ValueError异常，要求初始化
        if not _do_init:
            raise ValueError(
                "`FlaxSpeechEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果解码器的跨注意力隐藏大小不为None，则进行以下检查
        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器的跨注意力隐藏大小不等于编码器的隐藏大小，则抛出ValueError异常
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 确保输入和输出的嵌入不是共享的
        config.tie_word_embeddings = False
        # 使用给定的配置和其他参数实例化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)

        # 如果未提供输入形状，则设定默认的输入形状
        if input_shape is None:
            # 语音编码器几乎总是对序列长度维度进行降采样
            encoder_input_length = 1024
            # 根据编码器的输出长度获取解码器的输入长度
            decoder_input_length = module._get_feat_extract_output_lengths(encoder_input_length)
            # 设置输入形状为((1, encoder_input_length), (1, decoder_input_length))
            input_shape = ((1, encoder_input_length), (1, decoder_input_length))

        # 调用父类的初始化方法，传递配置、模块、输入形状、随机种子、数据类型和是否初始化的标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化权重函数，用于模型参数的初始化
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 解析输入形状，分为编码器和解码器的输入形状
        encoder_input_shape, decoder_input_shape = input_shape

        # 初始化编码器的输入数据为全零数组，数据类型为32位浮点数
        inputs = jnp.zeros(encoder_input_shape, dtype="f4")
        # 初始化编码器的注意力掩码为全1数组，数据类型为32位整数
        attention_mask = jnp.ones_like(inputs, dtype="i4")
        # 初始化解码器的输入标识为全零数组，数据类型为32位整数
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")
        # 初始化解码器的注意力掩码为与解码器输入标识相同形状的全1数组
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 获取输入数据的批量大小和序列长度
        batch_size, sequence_length = inputs.shape

        # 获取解码器输入标识的批量大小和序列长度
        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape
        # 检查编码器和解码器的批量大小是否相同，如果不同则抛出数值错误异常
        if not decoder_batch_size == batch_size:
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder"
                f" and {decoder_batch_size} for decoder."
            )
        
        # 根据解码器序列长度广播生成解码器的位置标识
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )

        # 使用随机数生成器分割生成参数随机种子和Dropout的随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        # 构建随机数种子字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模型的初始化方法初始化随机参数
        random_params = self.module.init(
            rngs,
            inputs,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
        )["params"]

        # 如果提供了预定义参数，则合并随机参数和预定义参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将预定义参数中缺失的键添加到随机参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结并返回合并后的参数字典
            return freeze(unflatten_dict(params))
        else:
            # 如果未提供预定义参数，则直接返回随机生成的参数字典
            return random_params
    # 初始化缓存，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存时的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 的隐藏状态序列，
                是编码器最后一层的输出的隐藏状态序列。在解码器的交叉注意力中使用。
        """
        # 初始化解码器的输入 ID，全为 1 的数组
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 解码器的注意力掩码，与输入 ID 形状相同，全为 1
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 解码器的位置 ID，广播到与输入 ID 形状相同的数组
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        # 定义一个内部函数来前向传播解码器模块
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        # 初始化模型变量，使用给定的参数和前向传播函数来获取缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器以初始化缓存
        )
        # 解冻缓存并返回
        return unfreeze(init_variables["cache"])

    # 获取特征提取器的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)

    # 编码器方法，根据给定的输入对语音编码器-解码器进行编码
    @add_start_docstrings(SPEECH_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def encode(
        self,
        inputs: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        freeze_feature_encoder: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import FlaxSpeechEncoderDecoderModel

        >>> # initialize a wav2vec2-2-bart from pretrained wav2vec2 and bart models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "facebook/wav2vec2-large-lv60", "facebook/bart-large"
        ... )

        >>> inputs = jnp.ones((2, 5000), dtype=jnp.float32)
        >>> encoder_outputs = model.encode(inputs)
        ```"""
        # 如果没有显式提供 output_attentions 参数，则使用配置文件中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式提供 output_hidden_states 参数，则使用配置文件中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式提供 return_dict 参数，则使用配置文件中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 attention_mask 参数为 None，则创建一个全 1 的掩码矩阵
        if attention_mask is None:
            attention_mask = jnp.ones_like(inputs, dtype="i4")

        # 处理可能存在的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义内部函数 _encoder_forward，用于执行编码器的前向传播
        def _encoder_forward(module, inputs, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(inputs, attention_mask, **kwargs)

        # 使用 Flax 模块的 apply 方法执行编码器的前向传播
        outputs = self.module.apply(
            {"params": params or self.params},
            inputs=jnp.array(inputs, dtype="f4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            freeze_feature_encoder=freeze_feature_encoder,
            rngs=rngs,
            method=_encoder_forward,
        )

        # 如果 return_dict 为 True，则构造一个 FlaxBaseModelOutput 对象
        if return_dict:
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # 返回模型的输出结果
        return outputs

    @add_start_docstrings(SPEECH_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        @add_start_docstrings_to_model_forward(SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING)
        ```
    # 使用装饰器替换返回文档字符串，指定输出类型为FlaxSeq2SeqLMOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义实例方法__call__，用于执行模型推理或训练
    def __call__(
        self,
        # 输入数据，一个NumPy数组
        inputs: jnp.ndarray,
        # 可选项，注意力掩码数组，用于指示输入中哪些元素是填充的
        attention_mask: Optional[jnp.ndarray] = None,
        # 可选项，解码器输入的ID数组，用于生成序列
        decoder_input_ids: Optional[jnp.ndarray] = None,
        # 可选项，解码器注意力掩码数组，用于指示解码器输入中哪些元素是填充的
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        # 可选项，解码器位置ID数组，指示每个解码器输入在序列中的位置
        decoder_position_ids: Optional[jnp.ndarray] = None,
        # 可选项，是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 可选项，是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 可选项，是否以字典形式返回输出结果
        return_dict: Optional[bool] = None,
        # 是否处于训练模式
        train: bool = False,
        # 是否冻结特征编码器
        freeze_feature_encoder: bool = False,
        # 模型参数的字典
        params: dict = None,
        # 随机数生成器的密钥
        dropout_rng: PRNGKey = None,
        ):
            r"""
            Returns:

            Examples:

            ```python
            >>> from transformers import FlaxSpeechEncoderDecoderModel, AutoTokenizer

            >>> # load a fine-tuned wav2vec2-2-bart model
            >>> model = FlaxSpeechEncoderDecoderModel.from_pretrained("patrickvonplaten/wav2vec2-2-bart-large")
            >>> # load output tokenizer
            >>> tokenizer_output = AutoTokenizer.from_pretrained("facebook/bart-large")

            >>> inputs = jnp.ones((2, 5000), dtype=jnp.float32)

            >>> # use bart's special bos, pad and eos tokens
            >>> model.config.decoder_start_token_id = model.decoder.config.bos_token_id
            >>> model.config.pad_token_id = model.decoder.config.pad_token_id
            >>> model.config.eos_token_id = model.decoder.config.eos_token_id

            >>> outputs = model.generate(inputs)
            # Assert something? More interesting input? dtype correct?
            ```
            """

            # Decide whether to use provided output attentions setting or default from model config
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # Decide whether to use provided output hidden states setting or default from model config
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # Decide whether to use provided return dict setting or default from model config
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # Prepare encoder inputs: if attention_mask is not provided, create one with all ones
            if attention_mask is None:
                attention_mask = jnp.ones_like(inputs, dtype="i4")

            # Prepare decoder inputs: decoder_input_ids cannot be None, raise error if so
            if decoder_input_ids is None:
                raise ValueError(
                    "`decoder_input_ids` cannot be `None`. For sequence to sequence training, `decoder_position_ids` must"
                    " be specified as an input argument."
                )
            # Prepare decoder attention mask: if not provided, create one with all ones
            if decoder_attention_mask is None:
                decoder_attention_mask = jnp.ones_like(decoder_input_ids)
            # Prepare decoder position ids: if not provided, broadcast from a range of sequence lengths
            if decoder_position_ids is None:
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )

            # Handle any dropout random number generator if provided
            rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

            # Apply the Flax module to the inputs and other provided arguments
            return self.module.apply(
                {"params": params or self.params},
                inputs=jnp.array(inputs, dtype="f4"),
                attention_mask=jnp.array(attention_mask, dtype="i4"),
                decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
                decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
                decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=not train,
                freeze_feature_encoder=freeze_feature_encoder,
                rngs=rngs,
            )
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        # 获取批量大小和解码器输入序列长度
        batch_size, seq_length = decoder_input_ids.shape

        # 使用初始化方法初始化缓存，获取过去的键值对
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 由于解码器使用因果掩码，可以创建一个静态的全1注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果存在解码器注意力掩码，则更新静态注意力掩码
        if decoder_attention_mask is not None:
            # 计算解码器位置ID，累积求和减一
            decoder_position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # 否则使用广播方式创建解码器位置ID
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        # 返回输入生成的字典，包括过去键值对、编码器输出、编码器注意力掩码、扩展后的解码器注意力掩码和解码器位置ID
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新输入以用于生成
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        **kwargs,
    ):
        # 从预训练的编码器-解码器模型中加载模型
        pass  # 这里省略了具体实现的注释，因为这个函数体没有具体代码，仅用于说明类方法的加载功能
```