# `.\models\vision_encoder_decoder\modeling_flax_vision_encoder_decoder.py`

```py
# 设定文件编码为 UTF-8
# 版权声明和许可证明，使用 Apache License, Version 2.0
#
# 导入必要的库和模块
import os
from typing import Optional, Tuple, Union

import flax.linen as nn  # 导入 Flax 的神经网络模块
import jax  # 导入 JAX，用于自动微分和加速计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的冻结字典相关功能
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 的字典扁平化和还原功能
from jax import lax  # 导入 JAX 的低级别 API
from jax.random import PRNGKey  # 导入 JAX 的随机数生成器 PRNGKey

# 导入模型输出相关的类
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
# 导入 FlaxPreTrainedModel 类，用于定义 Flax 模型的基类
from ...modeling_flax_utils import FlaxPreTrainedModel
# 导入文档字符串相关的工具函数和日志记录
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入自动配置类 AutoConfig
from ..auto.configuration_auto import AutoConfig
# 导入自动化模型类，包括通用模型和用于因果语言建模的模型
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
# 导入视觉编码解码配置类 VisionEncoderDecoderConfig
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个文档字符串常量，描述视觉-编码器-文本-解码器架构的类
_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical
    character recognition (OCR) yields a significant performance improvement.

    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any
    other models (see the examples for more information).
"""
    # 这个模型继承自 `FlaxPreTrainedModel`。查看超类文档以了解库实现的通用方法，如下载或保存模型、调整输入嵌入的大小、修剪头部等。

    # 这个模型还是一个 Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 子类。
    # 将其用作常规的 Flax Module，并参考 Flax 文档以获取与一般使用和行为相关的所有信息。

    # Parameters:
    #     config ([`VisionEncoderDecoderConfig`]): 模型配置类，包含模型的所有参数。
    #         使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    #     dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    #         计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）和 `jax.numpy.bfloat16`（在 TPU 上）之一。
    #
    #         这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 执行。
    #
    #         **注意，这只指定计算的数据类型，不影响模型参数的数据类型。**
    #
    #         如果您希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using the vision model's image processor. For example, using
            [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`] for details.
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
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

VISION_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using the vision model's image processor. For example, using
            [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxBaseModelOutput`] instead of a plain tuple.
"""

VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary. These tokens are generated by the model during
            decoding based on the provided `decoder_start_token_id`.
        decoder_start_token_id (`int`):
            The id of the token to start decoding with. This is usually the beginning-of-sequence token.
        encoder_outputs (`Union[FlaxBaseModelOutput, Tuple[jnp.ndarray]]`):
            Tuple comprising various elements depending on the configuration and inputs: logits as a jnp.ndarray of
            shape `(batch_size, sequence_length, vocab_size)`, hidden_states as a tuple of length `num_layers` with
            each element being a jnp.ndarray of shape `(batch_size, sequence_length, hidden_size)`, attentions as a
            tuple of length `num_layers` with each element being a jnp.ndarray of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`, and others.
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            1's in positions corresponding to input tokens to ignore and 0's in positions corresponding to input tokens
            to attend to. It's used to mask pad tokens in input sentences. It's also used to indicate the position of
            input tokens.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.
"""
    # 定义函数参数和其类型，以下是函数的解释说明
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            解码器输入序列标记的索引，大小为`(batch_size, target_sequence_length)`，可选。
            可以使用[`PreTrainedTokenizer`]获取这些索引。详见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。
            [什么是解码器输入 ID？](../glossary#decoder-input-ids)
            如果使用了 `past_key_values`，则可选地只需输入最后的 `decoder_input_ids`（参见 `past_key_values`）。
            对于序列到序列的训练，应提供 `decoder_input_ids`。如果没有提供 `decoder_input_ids`，模型将通过将 `input_ids` 向右移动创建此张量，用于去噪预训练。
    encoder_outputs (`tuple(tuple(jnp.ndarray)`):
        元组由 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`) 组成。
        `last_hidden_state` 大小为 `(batch_size, sequence_length, hidden_size)`，*可选*，是编码器最后一层的隐藏状态输出序列。用于解码器的交叉注意力。
    decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
        默认行为：生成一个张量，忽略 `decoder_input_ids` 中的填充标记。因果蒙版也将默认使用。
    decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        每个解码器输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.decoder.max_position_embeddings - 1]`。
    past_key_values (`Dict[str, jnp.ndarray]`, *optional*, 由 `init_cache` 返回或在传递先前的 `past_key_values` 时返回):
        预计算的隐藏状态字典（键和值在注意力块中）。可用于快速自回归解码。预计算的键和值隐藏状态的形状为 *[batch_size, max_length]*。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
    return_dict (`bool`, *optional*):
        如果设置为 `True`，模型将返回 [`~utils.FlaxCausalLMOutputWithCrossAttentions`] 而不是普通元组。
"""
# 定义一个 Flax 模型类，用于视觉编码器解码器
class FlaxVisionEncoderDecoderModule(nn.Module):
    # 模型配置信息，包括编码器和解码器配置
    config: VisionEncoderDecoderConfig
    # 默认数据类型为 JAX 的 float32 类型
    dtype: jnp.dtype = jnp.float32

    # 模型设置函数，用于初始化模型
    def setup(self):
        # 获取编码器和解码器的配置信息
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder

        # 从 `modeling_hybrid_clip.py` 中复制代码，并进行修改
        # 导入模型映射表，用于根据配置选择相应的编码器和解码器模块
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING

        # 根据编码器配置选择对应的编码器模块类
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        # 根据解码器配置选择对应的解码器模块类
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        # 使用选定的编码器模块和解码器模块初始化实例
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # 如果编码器的隐藏状态大小与解码器不同，并且解码器的交叉注意力隐藏大小为 None
        # 则需要进行编码器到解码器投影
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # 定义一个全连接层，用于编码器到解码器的投影
            self.enc_to_dec_proj = nn.Dense(
                self.decoder.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range),
                dtype=self.dtype,
            )
        else:
            # 否则不需要投影，设为 None
            self.enc_to_dec_proj = None

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取投影层模块的方法
    def _get_projection_module(self):
        return self.enc_to_dec_proj

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder

    # 模型调用函数，用于执行模型推理或训练
    def __call__(
        self,
        pixel_values,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        ):
            # 调用编码器（Encoder）模型，传入像素值、是否输出注意力权重、隐藏状态等参数，返回编码器的输出
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            # 获取编码器的隐藏状态
            encoder_hidden_states = encoder_outputs[0]

            # 如果存在编码器到解码器的投影层，则将编码器隐藏状态投影到解码器空间
            if self.enc_to_dec_proj is not None:
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            # 显式设置编码器的注意力掩码，全为1的矩阵
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

            # 调用解码器（Decoder）模型，传入解码器输入的token IDs、注意力掩码、位置 IDs、编码器隐藏状态及注意力掩码等参数，返回解码器的输出
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

            # 如果不返回字典形式的输出，则将解码器输出和编码器输出拼接起来返回
            if not return_dict:
                return decoder_outputs + encoder_outputs

            # 返回经过FlaxSeq2SeqLMOutput包装后的解码器输出和编码器输出
            return FlaxSeq2SeqLMOutput(
                logits=decoder_outputs.logits,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
# 使用装饰器将以下类添加文档字符串，该文档字符串与VISION_ENCODER_DECODER_START_DOCSTRING相符
@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
# 定义FlaxVisionEncoderDecoderModel类，继承自FlaxPreTrainedModel
class FlaxVisionEncoderDecoderModel(FlaxPreTrainedModel):
    # 类的文档字符串，描述了FlaxVisionEncoderDecoderModel的通用模型类特性
    r"""
    [`FlaxVisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture
    with the module (flax.nn.Module) of one of the base vision model classes of the library as encoder module and
    another one as decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method
    for the encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # 类属性，指定配置类为VisionEncoderDecoderConfig
    config_class = VisionEncoderDecoderConfig
    # 类属性，指定基础模型的前缀为"vision_encoder_decoder"
    base_model_prefix = "vision_encoder_decoder"
    # 类属性，主输入的名称为"pixel_values"
    main_input_name = "pixel_values"
    # 类属性，模块类为FlaxVisionEncoderDecoderModule
    module_class = FlaxVisionEncoderDecoderModule

    # 初始化方法
    def __init__(
        self,
        config: VisionEncoderDecoderConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果_do_init为False，则引发值错误，要求初始化为True
        if not _do_init:
            raise ValueError(
                "`FlaxVisionEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果未提供input_shape，则根据config.encoder的num_channels和image_size设置默认输入形状
        if input_shape is None:
            num_channels = getattr(config.encoder, "num_channels", 3)
            input_shape = (
                (1, config.encoder.image_size, config.encoder.image_size, num_channels),
                (1, 1),
            )

        # 如果decoder的cross_attention_hidden_size不为None，则验证其与encoder的hidden_size是否相等
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 使用给定的配置和其他参数实例化模块类，得到module对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类(FlaxPreTrainedModel)的初始化方法，传递配置、模块对象、输入形状、种子、数据类型和是否初始化标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化权重方法，使用给定的随机数生成器 rng，输入形状 input_shape 和可选的参数字典 params，返回冻结的参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 解包输入形状为编码器和解码器的输入形状
        encoder_input_shape, decoder_input_shape = input_shape

        # 初始化输入张量
        pixel_values = jnp.zeros(encoder_input_shape, dtype=self.dtype)  # 初始化像素值张量
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")  # 初始化解码器输入的 ID 张量
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)  # 初始化解码器注意力掩码张量

        # 检查批处理大小是否一致
        batch_size, _, _, _ = pixel_values.shape
        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape
        if not decoder_batch_size == batch_size:
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder "
                f"and {decoder_batch_size} for decoder."
            )

        # 创建解码器位置 ID 张量，广播到与解码器批处理大小和序列长度相匹配
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )

        # 拆分随机数生成器 rng，以用于参数和 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化随机参数
        random_params = self.module.init(
            rngs,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
        )["params"]

        # 如果提供了参数字典 params，则用随机初始化的参数覆盖缺失的参数键
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结并返回参数字典
            return freeze(unflatten_dict(params))
        else:
            # 否则，直接返回随机初始化的参数字典
            return random_params
    @add_start_docstrings(VISION_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def encode(
        self,
        pixel_values: jnp.ndarray,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,


        r"""
        Args:
            pixel_values (`jnp.ndarray`):
                Pixel values of the input images. A tensor of shape `(batch_size, channels, height, width)`.
            output_attentions (`Optional[bool]`, optional):
                Whether to return attentions weights. Defaults to `None`.
            output_hidden_states (`Optional[bool]`, optional):
                Whether to return hidden states. Defaults to `None`.
            return_dict (`Optional[bool]`, optional):
                Whether to return a dictionary instead of a tuple of outputs. Defaults to `None`.
            train (`bool`, optional):
                Whether in training mode. Defaults to `False`.
            params (`dict`, optional):
                Optional parameters for the encoding process. Defaults to `None`.
            dropout_rng (`PRNGKey`, optional):
                Random number generator key for dropout. Defaults to `None`.
        """
        ):
        r"""
        Returns:

        Example:

        ```
        >>> from transformers import AutoImageProcessor, FlaxVisionEncoderDecoderModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> # initialize a vit-gpt2 from pretrained ViT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "openai-community/gpt2"
        ... )

        >>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values
        >>> encoder_outputs = model.encode(pixel_values)
        ```
        """
        # Determine whether to output attentions, hidden states, and return as a dictionary based on inputs or default model config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Transpose pixel_values from channel last format to channel first format as expected by FlaxViTModel
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle random number generator states for dropout if specified
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # Define a function to perform the forward pass through the encoder module
        def _encoder_forward(module, pixel_values, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(pixel_values, **kwargs)

        # Apply the model's forward pass method with specified parameters and options
        outputs = self.module.apply(
            {"params": params or self.params},
            pixel_values=jnp.array(pixel_values, dtype=self.dtype),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

        # If return_dict is True, wrap outputs in FlaxBaseModelOutput format
        if return_dict:
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Return the processed outputs
        return outputs

    @add_start_docstrings(VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
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
        # 在此方法中解码模型的输出
        # 参数说明:
        # - decoder_input_ids: 解码器输入的标识符
        # - encoder_outputs: 编码器的输出
        # - decoder_attention_mask: 解码器的注意力掩码，可选
        # - decoder_position_ids: 解码器的位置标识符，可选
        # - past_key_values: 过去的键值对，用于缓存解码器状态，可选
        # - output_attentions: 是否输出注意力权重，可选
        # - output_hidden_states: 是否输出隐藏状态，可选
        # - return_dict: 是否返回字典形式的输出，可选
        # - train: 是否为训练模式
        # - params: 额外的参数字典，可选
        # - dropout_rng: 随机数生成器密钥，用于dropout操作，可选
        pass  # 实际的解码逻辑需要根据具体模型来实现

    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def __call__(
        self,
        pixel_values: jnp.ndarray,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        # 调用模型，实现前向传播
        # 参数说明同上述的decode方法
        pass  # 实际的前向传播逻辑需要根据具体模型来实现

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 准备用于生成的输入数据
        # 参数说明:
        # - decoder_input_ids: 解码器的输入标识符
        # - max_length: 生成的最大长度
        # - decoder_attention_mask: 解码器的注意力掩码，可选
        # - encoder_outputs: 编码器的输出，可选
        # - **kwargs: 其他关键字参数
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 创建扩展的注意力掩码，用于生成
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
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成过程中的输入
        # 参数说明:
        # - model_outputs: 模型的输出，包含过去的键值对
        # - model_kwargs: 模型调用的关键字参数
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    @classmethod
    # 定义一个类方法，用于从预训练的编码器和解码器模型中加载模型
    def from_encoder_decoder_pretrained(
        cls,
        # 可选参数：编码器预训练模型的名称或路径，可以是字符串或操作系统路径
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        # 可选参数：解码器预训练模型的名称或路径，可以是字符串或操作系统路径
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        # *model_args: 可变位置参数列表，用于接收额外的模型参数
        *model_args,
        # **kwargs: 可变关键字参数，用于接收额外的关键字参数
        **kwargs,
```