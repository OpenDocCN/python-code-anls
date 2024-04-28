# `.\transformers\models\vision_encoder_decoder\modeling_flax_vision_encoder_decoder.py`

```
# 该模块实现了基于 Vision-Encoder-Text-Decoder 架构的模型
# 该架构可使用任何预训练的视觉自编码模型作为编码器，
# 任何预训练的文本自回归模型作为解码器
# 自注意力层会自动添加到解码器中，需要在下游生成任务上进行微调

# 导入必要的模块和函数
import os
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

# 导入 Flax 模型所需的输出类型
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel
# 导入其他辅助函数和类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 设置文档开头
_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    这个类可用于初始化一个图像到文本序列的模型，
    可以使用任何预训练的视觉自编码模型作为编码器，
    任何预训练的文本自回归模型作为解码器。
    编码器是通过 `~AutoModel.from_pretrained` 函数加载的，
    解码器是通过 `~AutoModelForCausalLM.from_pretrained` 函数加载的。
    交叉注意力层会自动添加到解码器中，需要在下游生成任务上进行微调。

    使用预训练检查点初始化序列到序列模型在序列生成任务中的有效性
    在 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 中有所展示。

    此外，在 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) 中
    展示了利用大型预训练视觉模型进行光学字符识别 (OCR) 可以显著提高性能。

    在训练/微调这种 Vision-Encoder-Text-Decoder 模型之后，
    它可以像其他模型一样保存/加载 (更多信息请参见示例)。
"""
    # 此模型继承自 `FlaxPreTrainedModel`。请查阅超类文档，了解库为所有模型实现的通用方法（例如下载或保存、调整输入嵌入大小、修剪头等）。
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)
    
    # 此模型也是一个 Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 的子类。可以将其用作常规的 Flax 模块，并参考 Flax 文档以获取有关一般用法和行为的所有信息。
    This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    
    # 参数：
    Parameters:
    
        # config ([`VisionEncoderDecoderConfig`]): 包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
    
        # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`): 计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）和 `jax.numpy.bfloat16`（在 TPU 上）之一。
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
    
            # 这可以用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果指定了，则所有计算将使用给定的 `dtype` 执行。
            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
    
            # 注意这只指定计算的数据类型，不影响模型参数的数据类型。
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
    
            # 如果想要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
# 定义了一个文档字符串，用于描述视觉编码器解码器的输入参数
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用视觉模型的图像处理器获取像素值。例如，使用[`AutoImageProcessor`]。有关详细信息，请参见[`ViTImageProcessor.__call__`]。
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            译码器输入序列标记的索引。可以使用[`PreTrainedTokenizer`]获取索引。有关详细信息，请参见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            默认行为：生成一个张量，忽略`decoder_input_ids`中的填充标记。默认情况下还将使用因果掩码。
        decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个译码器输入序列标记在位置嵌入中的位置索引。在范围`[0, config.decoder.max_position_embeddings - 1]`中选择。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            如果设置为`True`，模型将返回一个[`~utils.FlaxSeq2SeqLMOutput`]而不是一个普通元组。
"""

# 定义了一个文档字符串，用于描述视觉编码器解码器的编码输入参数
VISION_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用视觉模型的图像处理器获取像素值。例如，使用[`AutoImageProcessor`]。有关详细信息，请参见[`ViTImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            如果设置为`True`，模型将返回一个[`~utils.FlaxBaseModelOutput`]而不是一个普通元组。
"""

# 定义了一个空的文档字符串，用于描述视觉编码器解码器的解码输入参数
VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 解码器输入序列标记在词汇表中的索引。
            # 可以使用 [`PreTrainedTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # 如果使用了 `past_key_values`，则可以选择只输入最后的 `decoder_input_ids`（参见 `past_key_values`）。
            # 对于序列到序列训练，应提供 `decoder_input_ids`。如果未提供 `decoder_input_ids`，模型将通过将 `input_ids` 向右移动来创建此张量以进行去噪预训练。
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            # 元组包含 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*optional*) 是编码器最后一层的隐藏状态序列。用于解码器的交叉注意力。
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 默认行为：生成一个张量，忽略 `decoder_input_ids` 中的填充标记。默认情况下还将使用因果掩码。
        decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 解码器输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.decoder.max_position_embeddings - 1]`。
        past_key_values (`Dict[str, jnp.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            # 预先计算的隐藏状态字典（注意力块中的键和值），可用于快速自回归解码。预先计算的键和值隐藏状态的形状为 *[batch_size, max_length]*。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 如果设置为 `True`，模型将返回一个 [`~utils.FlaxCausalLMOutputWithCrossAttentions`] 而不是一个普通元组。
# 定义一个FlaxVisionEncoderDecoderModule类，继承自nn.Module
class FlaxVisionEncoderDecoderModule(nn.Module):
    # 定义config属性为VisionEncoderDecoderConfig类型
    config: VisionEncoderDecoderConfig
    # 定义dtype属性为jnp.float32类型
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法
    def setup(self):
        # 获取encoder和decoder的配置
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder

        # 从`modeling_hybrid_clip.py`中复制代码，并进行修改
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING

        # 根据encoder和decoder的配置获取对应的模块类
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class

        # 创建encoder和decoder对象
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)

        # 如果encoder输出的维度与decoder的隐藏层维度不同，并且decoder的交叉注意力隐藏层维度为None
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # 创建一个Dense层，用于将encoder输出投影到decoder的隐藏层维度
            self.enc_to_dec_proj = nn.Dense(
                self.decoder.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range),
                dtype=self.dtype,
            )
        else:
            self.enc_to_dec_proj = None

    # 获取encoder模块
    def _get_encoder_module(self):
        return self.encoder

    # 获取投影模块
    def _get_projection_module(self):
        return self.enc_to_dec_proj

    # 获取decoder模块
    def _get_decoder_module(self):
        return self.decoder

    # 定义__call__方法，用于模型推理
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
        # 调用编码器模型，传入像素数值、是否输出注意力、是否输出隐藏状态、是否返回字典、是否确定性
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取编码器的隐藏状态
        encoder_hidden_states = encoder_outputs[0]

        # 可选地投影编码器隐藏状态
        if self.enc_to_dec_proj is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # 明确设置这个变量的优势是 TPU XLA 编译器尽早知道这个变量的形状，可以更好地优化。
        # 当 JIT 模型时，传递 `None` 可能会导致一些问题。
        # 在 Flax/JAX 中，我们只想为非张量函数输入传递 `None`。对于所有张量函数输入，我们应该始终传递张量而不是 `None`。
        batch_size, sequence_length = encoder_hidden_states.shape[:2]
        encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # 调用解码器模型，传入解码器输入 ID、解码器注意力掩码、解码器位置 ID、编码器隐藏状态、编码器注意力掩码、是否输出注意力、是否输出隐藏状态、是否返回字典、是否确定性
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

        # 如果不返回字典，则返回解码器输出和编码器输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回 FlaxSeq2SeqLMOutput 对象，包含解码器输出的 logits、隐藏状态、注意力、交叉注意力，以及编码器输出的最后隐藏状态、隐藏状态、注意力
        return FlaxSeq2SeqLMOutput(
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 导入必要的库
@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
# 定义一个类，继承自FlaxPreTrainedModel，用于实例化一个transformer架构的模型
class FlaxVisionEncoderDecoderModel(FlaxPreTrainedModel):
    r"""
    [`FlaxVisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture
    with the module (flax.nn.Module) of one of the base vision model classes of the library as encoder module and
    another one as decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method
    for the encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # 设置配置类为VisionEncoderDecoderConfig
    config_class = VisionEncoderDecoderConfig
    # 设置基础模型前缀为"vision_encoder_decoder"
    base_model_prefix = "vision_encoder_decoder"
    # 设置主输入名称为"pixel_values"
    main_input_name = "pixel_values"
    # 设置模块类为FlaxVisionEncoderDecoderModule
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
        # 如果_do_init为False，则抛出数值错误
        if not _do_init:
            raise ValueError(
                "`FlaxVisionEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果输入形状为None，则根据配置设置默认输入形状
        if input_shape is None:
            num_channels = getattr(config.encoder, "num_channels", 3)
            input_shape = (
                (1, config.encoder.image_size, config.encoder.image_size, num_channels),
                (1, 1),
            )

        # 如果解码器的交叉注意力隐藏大小不为None
        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器的交叉注意力隐藏大小不等于编码器的隐藏大小，则抛出数值错误
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 创建模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 解包输入形状
        encoder_input_shape, decoder_input_shape = input_shape

        # 初始化输入张量
        pixel_values = jnp.zeros(encoder_input_shape, dtype=self.dtype)
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 获取批量大小
        batch_size, _, _, _ = pixel_values.shape
        decoder_batch_size, decoder_sequence_length = decoder_input_ids.shape
        # 检查编码器和解码器的批量大小是否相同
        if not decoder_batch_size == batch_size:
            raise ValueError(
                f"The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder "
                f"and {decoder_batch_size} for decoder."
            )
        # 生成解码器位置编码
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length)
        )

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模型参数
        random_params = self.module.init(
            rngs,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
        )["params"]

        # 如果提供了预训练参数，则使用预训练参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    # 初始化缓存，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*）是编码器最后一层的隐藏状态输出。
                在解码器的交叉注意力中使用。
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

        # 初始化变量以检索缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
        )
        # 返回解冻后的缓存
        return unfreeze(init_variables["cache"])

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
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, FlaxVisionEncoderDecoderModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> # initialize a vit-gpt2 from pretrained ViT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "gpt2"
        ... )

        >>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values
        >>> encoder_outputs = model.encode(pixel_values)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # `FlaxViTModel` expects channel first format, but `FlaxViTModule` expects channel last format.
        # Currently, we assume this holds for all Flax vision models, and perform a transpose here.
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # Define a function to forward pass through the encoder module
        def _encoder_forward(module, pixel_values, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(pixel_values, **kwargs)

        # Apply the forward pass through the module
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

        # If return_dict is True, create a FlaxBaseModelOutput object
        if return_dict:
            outputs = FlaxBaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @add_start_docstrings(VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义一个解码方法，用于将编码器输出解码为目标序列
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
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义调用方法，用于模型的前向传播
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
    # 准备生成输入，用于生成目标序列
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape

        # 初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        
        # 创建扩展的注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果存在解码器注意力掩码，则更新注意力掩码和位置ID
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

    # 更新生成输入，用于生成目标序列
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    # 类方法
    @classmethod
    # 从预训练的编码器和解码器模型中创建一个新的实例
    def from_encoder_decoder_pretrained(
        cls,
        # 编码器预训练模型的名称或路径，可选参数
        encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        # 解码器预训练模型的名称或路径，可选参数
        decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        # 其他模型参数
        *model_args,
        # 其他关键字参数
        **kwargs,
```