# `.\models\encoder_decoder\modeling_flax_encoder_decoder.py`

```
# coding=utf-8
# 设置文件编码为UTF-8，支持多种语言字符的正确显示和处理

# Copyright 2021 The HuggingFace Inc. team.
# 版权声明，归属于HuggingFace团队，2021年

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证版本2.0进行授权

# you may not use this file except in compliance with the License.
# 使用此文件需遵守Apache许可证

# You may obtain a copy of the License at
# 可以在以下链接获取许可证副本

# http://www.apache.org/licenses/LICENSE-2.0
# 许可证链接

# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，

# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证分发的软件提供“按原样”

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的明示或暗示的保证或条件。

# See the License for the specific language governing permissions and
# 许可证中规定了相关权限的管理规则

# limitations under the License.
# 和使用限制。

""" Classes to support Flax Encoder-Decoder architectures"""
# 文档字符串，说明这里包含支持Flax编解码器架构的类


import os
# 导入os模块，用于操作操作系统功能

from typing import Optional, Tuple, Union
# 导入类型标注模块，用于声明变量类型

import flax.linen as nn
# 导入flax.linen模块，主要用于构建神经网络模型

import jax
# 导入jax模块，用于高效数值计算和自动微分

import jax.numpy as jnp
# 导入jax.numpy模块，用于数值计算，类似numpy但优化为使用在jax

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# 导入FrozenDict, freeze, unfreeze，用于处理不可变字典

from flax.traverse_util import flatten_dict, unflatten_dict
# 导入flatten_dict, unflatten_dict，用于字典结构的展平和还原

from jax import lax
# 导入jax中的lax模块，用于控制流和其他低级操作

from jax.random import PRNGKey
# 导入PRNGKey，用于生成伪随机数生成器的密钥

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
# 从相对路径导入Flax模型输出相关类

from ...modeling_flax_utils import FlaxPreTrainedModel
# 从相对路径导入Flax预训练模型基类

from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从相对路径导入工具函数和日志模块

from ..auto.configuration_auto import AutoConfig
# 从相对路径导入自动配置处理类

from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
# 从相对路径导入Flax自动模型类和因果语言模型

from .configuration_encoder_decoder import EncoderDecoderConfig
# 从当前路径导入编解码器配置类


logger = logging.get_logger(__name__)
# 初始化日志对象，用于当前文件的日志记录

_CONFIG_FOR_DOC = "EncoderDecoderConfig"
# 定义文档中使用的配置名变量，指向EncoderDecoderConfig类

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like
    # 此代码段为模型的参数说明，包括了配置和数据类型
    # config: 模型配置类，包含模型的所有参数。使用配置文件初始化模型不会加载与模型相关的权重，只加载配置。可以使用~FlaxPreTrainedModel.from_pretrained方法加载模型权重。
    # dtype: 计算的数据类型，默认为jax.numpy.float32。可以是jax.numpy.float32、jax.numpy.float16（在GPU上）、jax.numpy.bfloat16（在TPU上）中的一种。
    #        可以用来在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，所有计算都将使用给定的dtype进行。
    #        注意，这只是指定计算的dtype，不影响模型参数的dtype。
    #        如果要更改模型参数的dtype，请参见~FlaxPreTrainedModel.to_fp16和~FlaxPreTrainedModel.to_bf16。
# 定义编码器-解码器模型的输入参数说明文档字符串
ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列令牌在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。在 `[0, 1]` 范围内选择的掩码值：

            - 对于**未被掩码**的标记，值为 1，
            - 对于**被掩码**的标记，值为 0。

            [什么是注意力掩码?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            解码器输入序列令牌在词汇表中的索引。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是解码器输入 ID?](../glossary#decoder-input-ids)

            对于序列到序列训练，应提供 `decoder_input_ids`。`decoder_input_ids` 应该在模型外部创建，通过将 `labels` 向右移动，
            将 -100 替换为 `pad_token_id`，并用 `decoder_start_token_id` 预置它们。
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            默认行为：生成一个张量，忽略 `decoder_input_ids` 中的填充标记。默认情况下还将使用因果掩码。
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列令牌在位置嵌入中的位置索引。在范围 `[0, config.encoder.max_position_embeddings - 1]` 中选择。
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个解码器输入序列令牌在位置嵌入中的位置索引。在范围 `[0, config.decoder.max_position_embeddings - 1]` 中选择。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            如果设置为 `True`，模型将返回一个 [`~utils.FlaxSeq2SeqLMOutput`] 而不是一个简���的元组。
"""

# 定义编码器-解码器模型编码输入参数说明文档字符串
ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
    # 定义函数输入参数
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记的索引。默认情况下，忽略填充。
    
            可以使用 [`PreTrainedTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
    
            [什么是输入ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免对填充的标记进行注意力计算的掩码。掩码的值范围为 `[0, 1]`：
    
            - 1 表示**不屏蔽**的标记，
            - 0 表示**屏蔽**的标记。
    
            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。取值范围为 `[0, config.encoder.max_position_embeddings - 1]`。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            如果设置为 `True`，模型将返回一个 [`~utils.FlaxBaseModelOutput`]，而不是一个普通的元组。
# 定义一个文档字符串常量，用于存储编码器解码器解码输入的文档字符串
ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = r"""
"""

# 定义一个名为FlaxEncoderDecoderModule的类，继承自nn.Module
class FlaxEncoderDecoderModule(nn.Module):
    config: EncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    # 设置模块
    def setup(self):
        encoder_config = self.config.encoder  # 获取编码器配置
        decoder_config = self.config.decoder  # 获取解码器配置

        # 从 `...models.auto.modeling_flax_auto` 中导入 `FLAX_MODEL_FOR_CAUSAL_LM_MAPPING`, `FLAX_MODEL_MAPPING` ，并做一些修改
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING

        # 获取编码器模块和解码器模块
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        # 初始化编码器和解码器实例
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # 如果编码器输出需要投影到不同的维度以供解码器使用
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # 初始化投影层
            self.enc_to_dec_proj = nn.Dense(
                self.decoder.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range),
                dtype=self.dtype,
            )
        else:
            # 否则，设置投影层为None
            self.enc_to_dec_proj = None

    # 返回编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 返回投影模块
    def _get_projection_module(self):
        return self.enc_to_dec_proj

    # 返回解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 定义__call__方法
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
        # 使用 self.encoder 处理输入数据，得到编码器的输出
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

        # 可选地对编码器的隐藏状态进行投影
        if self.enc_to_dec_proj is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # 使用 self.decoder 处理解码器的输入数据，得到解码器的输出
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

        # 如果 return_dict 为 True，则返回格式化的输出，包括解码器和编码器的输出
        return FlaxSeq2SeqLMOutput(
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 使用 `add_start_docstrings` 装饰器，添加文档字符串到类中
@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
# 定义 `FlaxEncoderDecoderModel` 类，继承自 `FlaxPreTrainedModel` 类
class FlaxEncoderDecoderModel(FlaxPreTrainedModel):
    # `FlaxEncoderDecoderModel` 类的文档字符串
    r"""
    [`FlaxEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    the module (flax.nn.Module) of one of the base model classes of the library as encoder module and another one as
    decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method for the
    encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # `config_class` 属性指向 `EncoderDecoderConfig` 类
    config_class = EncoderDecoderConfig
    # `base_model_prefix` 属性值为 "encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # `module_class` 属性指向 `FlaxEncoderDecoderModule` 类
    module_class = FlaxEncoderDecoderModule

    # 定义初始化方法
    def __init__(
        self,
        config: EncoderDecoderConfig,  # 接收名为 `config` 的参数，类型为 `EncoderDecoderConfig` 类型
        input_shape: Optional[Tuple] = None,  # 可选参数 `input_shape`，类型为元组，初始值为 `None`
        seed: int = 0,  # 整数参数 `seed`，初始值为 0
        dtype: jnp.dtype = jnp.float32,  # 参数 `dtype`，类型为 `jnp.dtype`，初始值为 `jnp.float32`
        _do_init: bool = True,  # 布尔类型参数 `_do_init`，初始值为 `True`
        **kwargs,  # 接收任意其他未命名参数
    ):
        # 如果 `input_shape` 为 `None`，则设定默认值 ((1, 1), (1, 1))
        if input_shape is None:
            input_shape = ((1, 1), (1, 1))

        # 若不进行初始化操作，则抛出 ValueError
        if not _do_init:
            raise ValueError(
                "`FlaxEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果解码器的 `cross_attention_hidden_size` 不为空，且不等于编码器的 `hidden_size`，则抛出 ValueError
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 使用 `FlaxEncoderDecoderModule` 类创建 `module` 对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，设置相关属性
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型的参数，包括输入形状、参数字典等
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 解压输入形状
        encoder_input_shape, decoder_input_shape = input_shape

        # 初始化输入张量
        input_ids = jnp.zeros(encoder_input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 获取批大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 创建位置编码
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 获取解码器批大小和序列长度
        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape
        # 检查编码器和解码器的批大小是否一致
        if not decoder_batch_size == batch_size:
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder"
                f" and {decoder_batch_size} for decoder."
            )
        # 创建解码器的位置编码
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )

        # 划分随机数源
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模型参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果有预定义的参数，则将其与随机参数合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs`包含(`last_hidden_state`，*可选*: `hidden_states`，*可选*: `attentions`)。`last_hidden_state`的形状为`(batch_size, sequence_length, hidden_size)`，*可选*)是编码器最后一层的输出的隐藏状态序列。用于解码器中的交叉注意力。

        """
        # 初始化输入变量以检索缓存
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        # 使用初始参数初始化模型，传入解码器输入、解码器注意力掩码、解码器位置编码、编码器隐状态以及指定需要初始化缓存的方法
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC)
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

        ```python
        >>> from transformers import FlaxEncoderDecoderModel, BertTokenizer

        >>> # initialize a bert2gpt2 from pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> input_ids = tokenizer.encode(text, return_tensors="np")
        >>> encoder_outputs = model.encode(input_ids)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果传入的 output_attentions 不为空，则使用之，否则使用配置中的 output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果传入的 output_hidden_states 不为空，则使用之，否则使用配置中的 output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 如果传入的 return_dict 不为空，则使用之，否则使用配置中的 return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果 attention_mask 为空，则构造一个形状与 input_ids 相同的全 1 矩阵作为默认的 attention_mask
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        
        # 处理任何 PRNG 如果需要
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 调用 self.module.apply 方法进行编码器的正向传播
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

        if return_dict:
            # 如果需要返回字典形式的结果，则构造一个 FlaxBaseModelOutput 对象
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @add_start_docstrings(ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 对解密过程进行定义，接受一系列输入参数
    def decode(
        self,
        decoder_input_ids,  # 解码器输入的token IDs
        encoder_outputs,  # 编码器的输出
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 解码器的注意力掩码
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 解码器位置 IDs
        past_key_values: dict = None,  # 过去的键值对
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        train: bool = False,  # 是否为训练模式
        params: dict = None,  # 参数
        dropout_rng: PRNGKey = None,  # 随机数产生器
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)  # 添加描述原始模型前向输入的注释
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)  # 替换返回值的描述字符串为提供的配置类
    # 调用方法，接受一系列输入参数
    def __call__(
        self,
        input_ids: jnp.ndarray,  # 输入 token IDs
        attention_mask: Optional[jnp.ndarray] = None,  # 注意力掩码
        decoder_input_ids: Optional[jnp.ndarray] = None,  # 解码器输入的 token IDs
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 解码器的注意力掩码
        position_ids: Optional[jnp.ndarray] = None,  # 位置 IDs
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 解码器位置 IDs
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        train: bool = False,  # 是否为训练模式
        params: dict = None,  # 参数
        dropout_rng: PRNGKey = None,  # 随机数产生器
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入 token IDs
        max_length,  # 最大长度
        attention_mask: Optional[jax.Array] = None,  # 注意力掩码
        decoder_attention_mask: Optional[jax.Array] = None,  # 解码器的注意力掩码
        encoder_outputs=None,  # 编码器的输出，默认为None
        **kwargs,  # 其它关键字参数
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        
        # 注意通常需要将注意力掩码中超过输入 token 数和小于缓存长度的位置填充0��但由于解码器使用因果掩码，这些位置已经被遮蔽
        # 因此我们可以在这里创建一个静态的注意力掩码，这对编译更有效率
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            decoder_position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,  # 过去的键值对
            "encoder_outputs": encoder_outputs,  # 编码器的输出
            "encoder_attention_mask": attention_mask,  # 编码器的注意力掩码
            "decoder_attention_mask": extended_attention_mask,  # 解码器的注意力掩码
            "decoder_position_ids": decoder_position_ids,  # 解码器位置 IDs
        }
    # 更新用于生成的输入参数
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 将模型输出的过去键值添加到模型参数中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新解码器的位置标识，将其设为原位置标识的最后一位加一
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        # 返回更新后的模型参数
        return model_kwargs

    # 从预训练的编码器和解码器中创建一个实例
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        **kwargs,
```