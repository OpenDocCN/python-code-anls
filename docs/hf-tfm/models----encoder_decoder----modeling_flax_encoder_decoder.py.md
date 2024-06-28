# `.\models\encoder_decoder\modeling_flax_encoder_decoder.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" Classes to support Flax Encoder-Decoder architectures"""


import os
from typing import Optional, Tuple, Union

import flax.linen as nn  # 导入 Flax 的 Linen 模块，用于定义神经网络结构
import jax  # 导入 JAX，用于自动求导和加速数值计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口，用于操作多维数组
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的冻结字典相关函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 的字典扁平化和反扁平化工具函数
from jax import lax  # 导入 JAX 的 lax 模块，提供了一些数值计算的基本操作
from jax.random import PRNGKey  # 导入 JAX 的随机数生成模块

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput  # 导入输出相关类
from ...modeling_flax_utils import FlaxPreTrainedModel  # 导入 Flax 预训练模型基类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具函数和日志记录器
from ..auto.configuration_auto import AutoConfig  # 导入自动配置类
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM  # 导入自动模型加载类
from .configuration_encoder_decoder import EncoderDecoderConfig  # 导入编码解码器配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    # 定义 EncoderDecoder 类，继承自 FlaxPreTrainedModel 类
    class EncoderDecoder(FlaxPreTrainedModel):
        # 初始化方法，根据给定的配置 config 初始化模型
        def __init__(self, config: EncoderDecoderConfig):
            # 调用父类的初始化方法，传入配置 config
            super().__init__(config)
    
        # forward 方法用于模型推理，接收输入并返回输出
        def forward(
            self,
            # 输入数据
            input_ids: jnp.ndarray,
            # 注意力掩码
            attention_mask: jnp.ndarray,
            # token 类型 IDs
            token_type_ids: jnp.ndarray = None,
            # 位置编码
            position_ids: jnp.ndarray = None,
            # 校准
            inputs_embeds: jnp.ndarray = None,
            # 输出模型
            output_attentions: bool = False,
            # 输出层
            output_hidden_states: bool = False,
            # 返回结果
            return_dict: bool = False,
        ) -> Union[FlaxBaseModelOutput, Tuple[jnp.ndarray]]:
            # 参数解释
            """
            forward方法用于模型推理，接收一系列输入数据并返回模型输出结果。
    
            Parameters:
                input_ids (jax.numpy.ndarray): 输入的 token IDs.
                attention_mask (jax.numpy.ndarray): 注意力掩码，用于指示哪些位置是 padding 的.
                token_type_ids (jax.numpy.ndarray, optional): token 类型 IDs，默认为 None.
                position_ids (jax.numpy.ndarray, optional): 位置编码，默认为 None.
                inputs_embeds (jax.numpy.ndarray, optional): 输入的嵌入向量，默认为 None.
                output_attentions (bool, optional): 是否输出注意力权重，默认为 False.
                output_hidden_states (bool, optional): 是否输出所有隐藏状态，默认为 False.
                return_dict (bool, optional): 是否返回字典格式的输出，默认为 False.
    
            Returns:
                Union[FlaxBaseModelOutput, Tuple[jax.numpy.ndarray]]: 模型输出结果，可能为多种格式的返回值。
            """
            # 实现模型的前向推理过程
            raise NotImplementedError
    
        # 静态方法，用于从预训练模型加载权重
        @classmethod
        def from_pretrained(
            cls,
            # 模型路径或标识符
            pretrained_model_name_or_path: str,
            # 模型配置
            config: Optional[EncoderDecoderConfig] = None,
            # 数据类型，默认为 float32
            dtype: Optional[jax.numpy.dtype] = jnp.float32,
            # 本地缓存目录
            local_files_only: bool = False,
            # 使用显存
            use_auth_token: Optional[Union[bool, str]] = None,
            # 一系列附加关键字参数
            **kwargs,
        ) -> "FlaxPreTrainedModel":
            # 参数解释
            """
            从预训练模型加载模型权重和配置信息。
    
            Parameters:
                pretrained_model_name_or_path (str): 预训练模型的路径或标识符.
                config (Optional[EncoderDecoderConfig]): 模型配置，可选.
                dtype (Optional[jax.numpy.dtype]): 计算时使用的数据类型，默认为 jax.numpy.float32.
                local_files_only (bool): 是否只使用本地文件，默认为 False.
                use_auth_token (Optional[Union[bool, str]]): 是否使用授权令牌，默认为 None.
                **kwargs: 其他关键字参数.
    
            Returns:
                FlaxPreTrainedModel: 加载并返回预训练模型.
            """
            # 如果未提供配置，创建一个空的配置对象
            if config is None:
                config = EncoderDecoderConfig()
    
            # 获取模型的 URL 或本地路径
            resolved_model_path = hf_cache_or_filename(pretrained_model_name_or_path, kwargs)
    
            # 从 URL 或本地路径加载模型文件
            model_file = download_model_from_path(resolved_model_path, local_files_only=local_files_only)
    
            # 加载模型的配置信息，这里只加载配置而不加载权重
            model_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
    
            # 如果指定了 dtype，则将模型的计算类型设置为给定的 dtype
            if dtype is not None:
                model_config.dtype = dtype
    
            # 根据配置创建模型实例
            model = cls(config=model_config, **kwargs)
    
            # 如果存在本地缓存，加载权重
            if os.path.isfile(model_file):
                # 使用 JAX 来加载权重
                model_params = load_flax_weights_in_model(model, model_file)
    
            # 返回加载好权重的模型实例
            return model
    
        # 方法用于将模型参数转换为半精度（float16）
        def to_fp16(self):
            # 参数解释
            """
            将模型参数转换为半精度（float16）.
    
            Returns:
                EncoderDecoder: 转换后的半精度模型实例.
            """
            # 实现方法体
            raise NotImplementedError
    
        # 方法用于将模型参数转换为 bfloat16
        def to_bf16(self):
            # 参数解释
            """
            将模型参数转换为 bfloat16.
    
            Returns:
                EncoderDecoder: 转换后的 bfloat16 模型实例.
            """
            # 实现方法体
            raise NotImplementedError
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For sequence to sequence training, `decoder_input_ids` should be provided. `decoder_input_ids` should be
            created outside of the model by shifting the `labels` to the right, replacing -100 by the `pad_token_id`
            and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.encoder.max_position_embeddings - 1]`.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.decoder.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.
"""

ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.encoder.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.
"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.encoder.max_position_embeddings - 1]`.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxBaseModelOutput`] instead of a plain tuple.
"""

ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = r"""
"""

# 定义一个 Flax 编码器解码器模块的类
class FlaxEncoderDecoderModule(nn.Module):
    # 类属性：配置信息为 EncoderDecoderConfig 类型，数据类型为 jnp.float32
    config: EncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 获取编码器和解码器的配置
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder

        # 从 modeling_flax_auto 模块导入 FLAX_MODEL_MAPPING 和 FLAX_MODEL_FOR_CAUSAL_LM_MAPPING
        # encoder_module 是根据 encoder_config 类型从 FLAX_MODEL_MAPPING 中获取的模块类
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        # decoder_module 是根据 decoder_config 类型从 FLAX_MODEL_FOR_CAUSAL_LM_MAPPING 中获取的模块类
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        # 使用 encoder_module 和 decoder_module 初始化编码器和解码器实例
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # 如果编码器输出的隐藏状态维度与解码器不同，并且解码器的交叉注意力隐藏状态尺寸为 None
        # 则定义一个线性层 enc_to_dec_proj，用于将编码器输出投影到解码器所需的隐藏状态维度
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Dense(
                self.decoder.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range),
                dtype=self.dtype,
            )
        else:
            self.enc_to_dec_proj = None

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取投影模块的方法
    def _get_projection_module(self):
        return self.enc_to_dec_proj

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder

    # 调用实例时的方法，用于执行编码解码过程
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        ):
            # 调用编码器模型，传入输入的编码器相关参数
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            # 获取编码器的隐藏状态
            encoder_hidden_states = encoder_outputs[0]

            # 可选地投影编码器的隐藏状态到解码器
            if self.enc_to_dec_proj is not None:
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            # 调用解码器模型，传入解码器相关参数以及编码器的隐藏状态
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            # 如果 return_dict 为 False，则返回解码器和编码器的输出
            if not return_dict:
                return decoder_outputs + encoder_outputs

            # 如果 return_dict 为 True，则返回 FlaxSeq2SeqLMOutput 对象，包含解码器的输出和编码器的相关信息
            return FlaxSeq2SeqLMOutput(
                logits=decoder_outputs.logits,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
# 使用装饰器向FlaxEncoderDecoderModel类添加文档字符串
@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
# 定义FlaxEncoderDecoderModel类，继承自FlaxPreTrainedModel
class FlaxEncoderDecoderModel(FlaxPreTrainedModel):
    """
    [`FlaxEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    the module (flax.nn.Module) of one of the base model classes of the library as encoder module and another one as
    decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method for the
    encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # 指定配置类为EncoderDecoderConfig
    config_class = EncoderDecoderConfig
    # 指定基础模型的前缀
    base_model_prefix = "encoder_decoder"
    # 指定模块类为FlaxEncoderDecoderModule
    module_class = FlaxEncoderDecoderModule

    # 初始化方法
    def __init__(
        self,
        config: EncoderDecoderConfig,           # 配置对象，类型为EncoderDecoderConfig
        input_shape: Optional[Tuple] = None,    # 输入形状，可选的元组
        seed: int = 0,                          # 随机种子，默认为0
        dtype: jnp.dtype = jnp.float32,         # 数据类型，默认为jnp.float32
        _do_init: bool = True,                  # 是否初始化的标志，默认为True
        **kwargs,                               # 其他关键字参数
    ):
        # 如果没有指定输入形状，则设置默认输入形状为((1, 1), (1, 1))
        if input_shape is None:
            input_shape = ((1, 1), (1, 1))

        # 如果_do_init为False，则抛出错误，不能创建未初始化的FlaxEncoderDecoderModel
        if not _do_init:
            raise ValueError(
                "`FlaxEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果配置中decoder的cross_attention_hidden_size不为None
        if config.decoder.cross_attention_hidden_size is not None:
            # 检查decoder的cross_attention_hidden_size是否等于encoder的hidden_size
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 使用配置和其他关键字参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类FlaxPreTrainedModel的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        encoder_input_shape, decoder_input_shape = input_shape  # 解包输入形状元组

        # 初始化编码器的输入张量
        input_ids = jnp.zeros(encoder_input_shape, dtype="i4")  # 创建全零的整数张量
        attention_mask = jnp.ones_like(input_ids)  # 创建与input_ids形状相同的全1张量作为注意力掩码

        # 初始化解码器的输入张量
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")  # 创建全零的整数张量
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)  # 创建与decoder_input_ids形状相同的全1张量作为注意力掩码

        batch_size, sequence_length = input_ids.shape  # 获取编码器输入张量的批量大小和序列长度
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))  # 根据序列长度广播位置编码

        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape  # 获取解码器输入张量的批量大小和序列长度
        if not decoder_batch_size == batch_size:  # 如果编码器和解码器的批量大小不相等，抛出值错误
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder"
                f" and {decoder_batch_size} for decoder."
            )
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )  # 根据解码器序列长度广播解码器的位置编码

        params_rng, dropout_rng = jax.random.split(rng)  # 使用随机数生成器拆分用于参数初始化和dropout的随机数种子
        rngs = {"params": params_rng, "dropout": dropout_rng}  # 组成随机数种子字典

        random_params = self.module.init(  # 使用模块的初始化方法初始化随机参数
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]  # 返回初始化后的参数

        if params is not None:  # 如果给定了预定义的参数
            random_params = flatten_dict(unfreeze(random_params))  # 展平和解冻随机参数
            params = flatten_dict(unfreeze(params))  # 展平和解冻预定义参数
            for missing_key in self._missing_keys:  # 对于每个缺失的键
                params[missing_key] = random_params[missing_key]  # 使用随机参数填充预定义参数的缺失部分
            self._missing_keys = set()  # 清空缺失键集合
            return freeze(unflatten_dict(params))  # 冻结和重构预定义参数并返回
        else:
            return random_params  # 否则直接返回随机初始化的参数
    # 初始化缓存函数，用于自动回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自动回归解码的批大小。定义了初始化缓存的批大小。
            max_length (`int`):
                自动回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态输出，
                在解码器的交叉注意力中使用。
        """
        # 初始化解码器输入的变量以检索缓存
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            # 获取解码器模块
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        # 使用解码器来初始化缓存，只需调用解码器来初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,
        )
        # 解冻并返回初始化的缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 编码函数，用于对输入进行编码
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        ):
        r"""
        Returns:

        Example:

        ```
        >>> from transformers import FlaxEncoderDecoderModel, BertTokenizer

        >>> # initialize a bert2gpt2 from pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "openai-community/gpt2")

        >>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> input_ids = tokenizer.encode(text, return_tensors="np")
        >>> encoder_outputs = model.encode(input_ids)
        ```"""
        # 设置输出注意力机制参数，若未指定则使用配置文件中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态参数，若未指定则使用配置文件中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典参数，若未指定则使用配置文件中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果没有提供注意力掩码，则创建一个与输入相同形状的全1注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果没有提供位置编码，则根据输入的长度广播生成位置编码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果需要处理任何伪随机数生成器（PRNG）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向传播函数
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            # 获取编码器模块
            encode_module = module._get_encoder_module()
            # 调用编码器模块进行编码
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 应用模型的前向传播，传入参数和配置
        outputs = self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

        # 如果需要返回字典，则构建相应的输出对象
        if return_dict:
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # 返回模型的输出结果
        return outputs

    # 添加开始的文档字符串注释，指定输入的解码器解码文档字符串
    @add_start_docstrings(ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    # 替换返回文档字符串，指定输出类型为带交叉注意力的FlaxCausalLMOutputWithCrossAttentions，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义一个解码方法，用于将编码器和解码器的输入转换为模型的输出
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
        # 这里使用了自定义的函数装饰器，为模型的前向传播添加了文档字符串
        @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
        def __call__(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            decoder_input_ids: Optional[jnp.ndarray] = None,
            decoder_attention_mask: Optional[jnp.ndarray] = None,
            position_ids: Optional[jnp.ndarray] = None,
            decoder_position_ids: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
        ):
            pass  # 此处省略了函数具体实现，由于是在类内部定义，可以访问类的其他成员变量和方法

    # 准备生成时的输入，初始化缓存和注意力掩码等
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存，通常用于存储解码器的过去键值对
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 创建一个扩展的注意力掩码，用于确保模型只关注当前生成位置之前的信息
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            # 根据解码器的注意力掩码动态更新扩展的注意力掩码
            decoder_position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # 如果没有提供解码器的注意力掩码，则使用默认的位置 IDs
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }
    # 更新生成过程中的模型参数
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 将模型输出中的过去键值添加到模型参数中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新解码器位置标识符，将其限制为最后一个位置加1
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        # 返回更新后的模型参数
        return model_kwargs

    # 从预训练的编码器-解码器模型中创建实例
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        **kwargs,
```