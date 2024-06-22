# `.\transformers\models\speech_encoder_decoder\modeling_flax_speech_encoder_decoder.py`

```py
# 设置编码，根据 UTF-8 编码规范
# 版权版本声明及其许可协议
# 如果没有许可协议，不得使用此文件，获取协议的副本，链接详细？
# 根据协议分发软件，采用“原样”（AS IS）的基础性分发，没有任何明示或暗示的保证和条件
# 请参考协议来规定特定语言的权限和限制
# ⠀
# 该类支持Flax语音编码器解码器体系结构
# 需要自动配置的配置



import os
# 执行模型输入输出操作
from typing import Optional, Tuple, Union

# Flax框架中的层
import flax.linen as nn
# 实现自动求导，计算张量等相关功能
import jax
import jax.numpy as jnp
# 冻结和解冻字典
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# 字典扁平化和反扁平化工具
from flax.traverse_util import flatten_dict, unflatten_dict
# 随机数生成器
from jax import lax
from jax.random import PRNGKey

# ---- 省略部分代码 ----

logger = logging.get_logger(__name__)

# 用于文档的配置
_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"
# 添加自起始docstrings
# 用于初始化语音序列到文本序列模型
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
    # 这个模型继承自FlaxPreTrainedModel。查看超类文档，了解库实现的所有模型的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。
    
    # 这个模型也是Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html)的子类。将其用作常规Flax模块，并参考Flax文档，了解与通用使用和行为相关的所有事项。
    
    # 参数：
    #    config ([`SpeechEncoderDecoderConfig`]): 具有模型所有参数的模型配置类。使用配置文件初始化不加载与模型相关的权重，只加载配置。查看[`~FlaxPreTrainedModel.from_pretrained`]方法，加载模型权重。
    #    dtype (`jax.numpy.dtype`, *optional*, 默认为`jax.numpy.float32`):计算的数据类型。可以是`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和`jax.numpy.bfloat16`（在TPU上）之一。
    #        这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，所有计算将使用给定的`dtype`执行。
    #        **请注意，这只指定计算的数据类型，不影响模型参数的数据类型。**
    #        如果要更改模型参数的数据类型，请参见[`~FlaxPreTrainedModel.to_fp16`]和[`~FlaxPreTrainedModel.to_bf16`]。
# 定义了一个名为SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING的文档字符串，说明了各种输入参数的形状和意义
    Args:
        inputs (`jnp.ndarray` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*):
            # 输入参数inputs的说明
            Float values of input raw speech waveform or speech features. Values can be obtained by loading a `.flac`
            or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile
            library (`pip install soundfile`). To prepare the array into `inputs`, either the [`Wav2Vec2Processor`] or
            [`Speech2TextProcessor`] should be used for padding and conversion into a tensor of type
            `torch.FloatTensor`.
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入参数attention_mask的说明
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 输入参数decoder_input_ids的说明
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
            # 输入参数decoder_attention_mask的说明
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入参数decoder_position_ids的说明
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.decoder.max_position_embeddings - 1]`.
        output_hidden_states (`bool`, *optional*):
            # 输入参数output_hidden_states的说明
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 输入参数return_dict的说明
            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.

SPEECH_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
# 定义了一个名为SPEECH_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING的文档字符串
    # 定义函数参数
    Args:
        # 输入参数，可以是原始语音波形或语音特征的浮点值数组
        inputs (`jnp.ndarray` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*):
            # 如果输入是 *.flac* 或 *.wav* 音频文件，可以通过 *List[float]* 或 *numpy.ndarray* 类型的数组加载到这里
            # 可以通过 soundfile 库 (*pip install soundfile*) 获得输入值数组
            # 为了准备输入数组 *inputs*，应该使用 [`Wav2Vec2Processor`] 或 [`Speech2TextProcessor`] 进行填充并将其转换为类型为 *torch.FloatTensor* 的张量
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 用于避免在填充标记索引上执行注意的遮罩。遮罩值选择于 `[0, 1]` 之间：

            # - 对于**未被遮罩**的标记，值为 1
            # - 对于**被遮罩**的标记，值为 0

            # [什么是注意力遮罩?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意层的注意张量。有关更多详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 如果设置为 `True`，模型将返回 [`~utils.FlaxBaseModelOutput`] 而不是普通元组。
"""
定义了一个字符串变量，用于存储文档字符串
"""
SPEECH_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = r"""
"""


class FlaxSpeechEncoderDecoderModule(nn.Module):
    """
    定义了一个名为FlaxSpeechEncoderDecoderModule的类，继承自nn.Module
    """
    config: SpeechEncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """
        初始化方法，设置编码器和解码器
        """
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder

        # Copied from `modeling_hybrid_clip.py` with modifications.
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING

        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # encoder outputs might need to be projected to different dimension for decoder
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

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        计算卷积层的输出长度
        """

        add_adapter = self.config.encoder.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.encoder.conv_kernel, self.config.encoder.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

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
            # 如果编码器输出为空，则使用编码器处理输入
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    inputs,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    deterministic=deterministic,
                    freeze_feature_encoder=freeze_feature_encoder,
                )

            # 获取编码器隐藏状态
            encoder_hidden_states = encoder_outputs[0]

            # 可选择地投影编码器隐藏状态
            if self.enc_to_dec_proj is not None:
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            # 计算正确的编码器注意力掩码
            if attention_mask is not None:
                encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                    encoder_hidden_states.shape[1], attention_mask
                )
            else:
                encoder_attention_mask = None

            # 使用 flax 脚本模块 modeling_flax_wav2vec2.py
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

            # 如果不返回字典，则返回解码器和编码器的输出
            if not return_dict:
                return decoder_outputs + encoder_outputs

            # 返回 FlaxSeq2SeqLMOutput 字典对象
            return FlaxSeq2SeqLMOutput(
                logits=decoder_outputs.logits,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_hidden_states,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
# 使用装饰器添加起始文档字符串到 `FlaxSpeechEncoderDecoderModel` 类
@add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
class FlaxSpeechEncoderDecoderModel(FlaxPreTrainedModel):
    r"""
    [`FlaxSpeechEncoderDecoderModel`] 是一个通用的模型类，当使用 :meth*~transformers.FlaxAutoModel.from_pretrained* 
    类方法为编码器和 :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* 类方法为解码器创建时，
    它将被实例化为一个具有库中一个基本模型类的模块（flax.nn.Module）作为编码器模块和另一个作为解码器模块的变压器架构。
    """

    # 配置类为 SpeechEncoderDecoderConfig
    config_class = SpeechEncoderDecoderConfig
    # 基础模型前缀为 "speech_encoder_decoder"
    base_model_prefix: str = "speech_encoder_decoder"
    # 模块类为 FlaxSpeechEncoderDecoderModule
    module_class = FlaxSpeechEncoderDecoderModule

    def __init__(
        self,
        config: SpeechEncoderDecoderConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果不进行初始化，则抛出 ValueError
        if not _do_init:
            raise ValueError(
                "`FlaxSpeechEncoderDecoderModel` 不能在不进行初始化的情况下创建，`_do_init` 必须为 `True`。"
            )

        # 如果解码器的交叉注意力隐藏大小不为 None
        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器的交叉注意力隐藏大小不等于编码器的隐藏大小，则抛出 ValueError
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "如果解码器的配置中指定了 `cross_attention_hidden_size`，它必须等于编码器的 `hidden_size`。"
                    f" 得到了 {config.decoder.cross_attention_hidden_size} 作为"
                    f" `config.decoder.cross_attention_hidden_size` 和 {config.encoder.hidden_size} 作为"
                    " `config.encoder.hidden_size`。"
                )

        # 确保输入和输出嵌入不被绑定
        config.tie_word_embeddings = False
        # 使用给定配置和参数创建模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)

        # 如果输入形状为 None，则设置默认形状
        if input_shape is None:
            # 语音编码器几乎总是将序列长度维度下采样
            encoder_input_length = 1024
            # 获取特征提取器的输出长度
            decoder_input_length = module._get_feat_extract_output_lengths(encoder_input_length)
            input_shape = ((1, encoder_input_length), (1, decoder_input_length))

        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型权重函数，接受随机数种子、输入形状和参数字典（可选），返回初始化后的参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 分别获取编码器和解码器的输入形状
        encoder_input_shape, decoder_input_shape = input_shape

        # 初始化编码器的输入和注意力掩码，均为零矩阵和全一矩阵
        inputs = jnp.zeros(encoder_input_shape, dtype="f4")
        attention_mask = jnp.ones_like(inputs, dtype="i4")
        # 初始化解码器的输入和注意力掩码，均为零矩阵和全一矩阵
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 获取输入数据的批量大小和序列长度
        batch_size, sequence_length = inputs.shape

        # 获取解码器输入数据的批量大小和序列长度
        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape
        # 检查编码器和解码器的批量大小是否一致，若不一致则抛出错误
        if not decoder_batch_size == batch_size:
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder"
                f" and {decoder_batch_size} for decoder."
            )
        # 根据解码器序列长度广播创建解码器位置编码
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )

        # 使用随机数种子分割为参数和 dropout 随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        # 将参数和 dropout 随机数种子封装成字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模型参数，如果提供了初始参数则使用提供的，否则随机初始化
        random_params = self.module.init(
            rngs,
            inputs,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
        )["params"]

        # 如果提供了初始参数，则将缺失的键填充为随机初始化的参数
        if params is not None:
            # 将参数字典展开为一维，并转换为可变字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 遍历缺失的键，将其对应的值从随机初始化的参数中取出填充
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            # 清空缺失的键集合
            self._missing_keys = set()
            # 将填充后的参数字典重新冻结为不可变字典
            return freeze(unflatten_dict(params))
        else:
            # 如果未提供初始参数，则直接返回随机初始化的参数字典
            return random_params
    # 初始化缓存，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小。定义了初始化缓存的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，
                *可选* 是编码器最后一层的隐藏状态的序列。用于解码器的交叉注意力。

        """
        # 初始化用于检索缓存的输入变量
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

        # 使用 self.module.init 方法初始化模型参数，用于获取缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需要调用解码器来初始化缓存
        )
        # 返回解冻后的缓存变量
        return unfreeze(init_variables["cache"])

    # 获取特征提取器的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)

    # 对输入进行编码
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
        ```py"""
        # 检查是否指定输出注意力值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否指定输出隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否指定是否返回字典形式结果
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果没有指定注意力掩码，则创建一个全是1的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(inputs, dtype="i4")

        # 处理需要的PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向传播函数
        def _encoder_forward(module, inputs, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(inputs, attention_mask, **kwargs)

        # 使用模型的apply方法进行前向传播计算
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

        # 如果需要字典形式的输出，则转换输出
        if return_dict:
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    # 解码器的decode方法
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
    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING)
    # 使用装饰器替换返回值的文档字符串为指定的输出类型，并指定配置类为给定配置的配置类
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个可调用的方法
    def __call__(
        self,
        # 输入数据的张量，应该是一个 Numpy 数组
        inputs: jnp.ndarray,
        # 注意力掩码，可选参数，默认为 None
        attention_mask: Optional[jnp.ndarray] = None,
        # 解码器输入的标识符，可选参数，默认为 None
        decoder_input_ids: Optional[jnp.ndarray] = None,
        # 解码器的注意力掩码，可选参数，默认为 None
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        # 解码器的位置标识符，可选参数，默认为 None
        decoder_position_ids: Optional[jnp.ndarray] = None,
        # 是否输出注意力权重矩阵，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典对象，可选参数，默认为 None
        return_dict: Optional[bool] = None,
        # 是否处于训练模式，布尔值，默认为 False
        train: bool = False,
        # 是否冻结特征编码器，布尔值，默认为 False
        freeze_feature_encoder: bool = False,
        # 模型参数，字典对象，默认为 None
        params: dict = None,
        # 用于丢弃的随机数生成器密钥，PRNGKey 对象，默认为 None
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
        ```py
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(inputs, dtype="i4")

        # prepare decoder inputs
        if decoder_input_ids is None:
            raise ValueError(
                "`decoder_input_ids` cannot be `None`. For sequence to sequence training, `decoder_position_ids` must"
                " be specified as an input argument."
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 应用模型的前向传播
        return self.module.apply(
            # 使用传入的参数或者当前模型的参数
            {"params": params or self.params},
            # 转换输入数据类型为 float32
            inputs=jnp.array(inputs, dtype="f4"),
            # 转换注意力掩码数据类型为 int32
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            # 转换解码器输入标识数据类型为 int32
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            # 转换解码器注意力掩码数据类型为 int32
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            # 转换解码器位置标识数据类型为 int32
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            # 设置输出注意力
            output_attentions=output_attentions,
            # 设置输出隐藏状态
            output_hidden_states=output_hidden_states,
            # 设置返回字典
            return_dict=return_dict,
            # 设置是否冻结特征编码器
            freeze_feature_encoder=freeze_feature_encoder,
            # 如果不是训练模式，则设置 PRNG 为确定性
            deterministic=not train,
            # 设置 PRNG
            rngs=rngs,
        )
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape

        # 使用初始化缓存函数初始化过去的键值
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 注意，通常情况下，我们需要为 x > input.shape[-1] 和 x < cache_length 的位置放置 0 在 attention_mask 中。
        # 但由于解码器使用因果蒙版，这些位置已经被屏蔽了。
        # 因此，我们可以在这里创建一个单独的静态 attention_mask，对编译更有效
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            decoder_position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
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

    # 更新用于生成的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    # 从编码器-解码器预训练模型中构建
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        **kwargs,
```