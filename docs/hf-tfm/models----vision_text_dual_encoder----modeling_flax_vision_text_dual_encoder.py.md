# `.\transformers\models\vision_text_dual_encoder\modeling_flax_vision_text_dual_encoder.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" Flax VisionTextDualEncoder model."""

# 导入必要的库和模块
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

# 导入自定义的模块和函数
from ...modeling_flax_utils import FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import add_start_docstrings, logging
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FLAX_MODEL_MAPPING, FlaxAutoModel
from ..clip.modeling_flax_clip import FlaxCLIPOutput, FlaxCLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "VisionTextDualEncoderConfig"

# Vision-Text Dual Encoder 模型的起始文档字符串
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~FlaxAutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.

    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.

    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

     This model is also a
     [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it
     as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
     behavior.

    Finally, this model supports inherent JAX features such as:
``` 
    # JIT（即时编译）：JAX 中的一种编译技术，可提高代码的执行效率
    # 自动微分：JAX 中的自动微分功能，用于计算梯度
    # 向量化：JAX 中的向量化操作，通过 vmap 函数实现
    # 并行化：JAX 中的并行化操作，通过 pmap 函数实现

    参数：
        config（[`VisionTextDualEncoderConfig`]）：模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
        dtype（`jax.numpy.dtype`，*可选*，默认为 `jax.numpy.float32`）：
            计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）和 `jax.numpy.bfloat16`（在 TPU 上）之一。

            这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定了数据类型，则所有计算将使用给定的 `dtype` 执行。

            **请注意，这仅指定计算的数据类型，不会影响模型参数的数据类型。**

            如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
# 定义了一个文档字符串，用于描述 VisionTextDualEncoder 模块的输入参数
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下会忽略填充。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]` 范围内：

            - 1 表示**未被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下会忽略填充。可以使用图像处理器获取像素值（例如，如果使用 ViT 作为编码器，则应使用 [`AutoImageProcessor`]）。有关详细信息，请参阅 [`ViTImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义了一个 FlaxVisionTextDualEncoderModule 类，继承自 nn.Module
class FlaxVisionTextDualEncoderModule(nn.Module):
    # 类型注解，指定了 VisionTextDualEncoderConfig 类型的 config 属性
    config: VisionTextDualEncoderConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 设置函数，用于初始化模型参数
    def setup(self):
        # 获取视觉配置和文本配置
        vision_config = self.config.vision_config
        text_config = self.config.text_config

        # 设置视觉嵌入维度、文本嵌入维度和投影维度
        self.vision_embed_dim = vision_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        self.projection_dim = self.config.projection_dim

        # 根据配置获取视觉模型和文本模型的类
        vision_module = FLAX_MODEL_MAPPING.get(self.config.vision_config.__class__, FlaxCLIPVisionModel).module_class
        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class

        # 初始化视觉模型和文本模型
        self.vision_model = vision_module(vision_config, dtype=self.dtype)
        self.text_model = text_module(text_config, dtype=self.dtype)

        # 初始化视觉投影层和文本投影层
        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.text_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )

        # 初始化logit缩放参数
        self.logit_scale = self.param(
            "logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []
        )

    # 定义模型调用函数，接收输入参数并返回模型输出
    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        # 如果 return_dict 不为 None，则使用传入的 return_dict，否则使用配置中的 return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 使用视觉模型处理像素值，返回视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 使用文本模型处理输入，返回文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取视觉输出中的图像嵌入
        image_embeds = vision_outputs[1]
        # 使用视觉投影层处理图像嵌入
        image_embeds = self.visual_projection(image_embeds)

        # 获取文本输出中的文本嵌入
        text_embeds = text_outputs[1]
        # 使用文本投影层处理文本嵌入
        text_embeds = self.text_projection(text_embeds)

        # 对图像嵌入进行归一化处理
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        # 对文本嵌入进行归一化处理
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # 计算余弦相似度作为 logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        # 如果 return_dict 为 False，则返回一组元组
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        # 如果 return_dict 为 True，则返回 FlaxCLIPOutput 对象
        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
# 添加起始文档字符串到 FlaxVisionTextDualEncoderModel 类
@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class FlaxVisionTextDualEncoderModel(FlaxPreTrainedModel):
    # 设置配置类为 VisionTextDualEncoderConfig
    config_class = VisionTextDualEncoderConfig
    # 设置模块类为 FlaxVisionTextDualEncoderModule
    module_class = FlaxVisionTextDualEncoderModule

    # 初始化方法
    def __init__(
        self,
        config: VisionTextDualEncoderConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果 _do_init 不为 True，则抛出 ValueError
        if not _do_init:
            raise ValueError(
                "`FlaxVisionTextDualEncoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果 input_shape 为 None，则设置默认值
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))

        # 创建模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    # 初始化权重方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        token_type_ids = jnp.ones_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)

        # 生成随机像素值
        pixel_values = jax.random.normal(rng, input_shape[1])

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块参数
        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids, token_type_ids)[
            "params"
        ]

        # 如果已有参数，则合并参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 调用方法
    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
            # 如果 output_attentions 为 None，则使用配置中的 output_attentions
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 为 None，则使用配置中的 output_hidden_states
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 return_dict 为 None，则使用配置中的 return_dict
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 调整像素值的维度顺序
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

            # 如果 position_ids 为 None，则根据 input_ids 的维度广播生成位置 id
            if position_ids is None:
                position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

            # 如果 token_type_ids 为 None，则使用与 input_ids 相同形状的全零数组
            if token_type_ids is None:
                token_type_ids = jnp.zeros_like(input_ids)

            # 如果 attention_mask 为 None，则使用与 input_ids 相同形状的全一数组
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)

            # 处理任何需要的 PRNG
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            # 调用模块的 apply 方法进行前向传播
            return self.module.apply(
                {"params": params or self.params},
                jnp.array(input_ids, dtype="i4"),
                jnp.array(pixel_values, dtype=jnp.float32),
                jnp.array(attention_mask, dtype="i4"),
                jnp.array(position_ids, dtype="i4"),
                jnp.array(token_type_ids, dtype="i4"),
                not train,
                output_attentions,
                output_hidden_states,
                return_dict,
                rngs=rngs,
            )

        # 获取文本特征的方法
        def get_text_features(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            token_type_ids=None,
            params: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train=False,
    ):
        r"""
        Args:
            input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)

        Returns:
            text_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of text model.
        """
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, token_type_ids, deterministic):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                deterministic=deterministic,
            )
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return text_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    def get_image_features(
        self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train=False
        ):
        r"""
        Args:
            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`ImageFeatureExtractionMixin`]. See [`ImageFeatureExtractionMixin.__call__`] for details.

        Returns:
            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of vision model.
        """

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            # 获取视觉模型的输出
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            # 获取池化后的输出
            pooled_output = vision_outputs[1]  # pooled_output
            # 应用投影层到池化输出，得到图像特征
            image_features = module.visual_projection(pooled_output)
            return image_features

        # 应用模型到输入数据，获取图像特征
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
# 定义 VisionTextDualEncoderModel 的文档字符串，包含返回值和示例
VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING = r"""
    Returns:

    Examples:

    ```python
    >>> from PIL import Image
   >>> import requests
   >>> import jax
   >>> from transformers import (
    ...     FlaxVisionTextDualEncoderModel,
    ...     VisionTextDualEncoderProcessor,
    ...     AutoImageProcessor,
    ...     AutoTokenizer,
    ... )

   >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   >>> image_processor = AutoImageProcesor.from_pretrained("google/vit-base-patch16-224")
   >>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
   >>> model = FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(
    ...     "google/vit-base-patch16-224", "bert-base-uncased"
    ... )

   >>> # contrastive training
   >>> urls = [
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
    ... ]
   >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
   >>> inputs = processor(
    ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="np", padding=True
    ... )
   >>> outputs = model(
    ...     input_ids=inputs.input_ids,
    ...     attention_mask=inputs.attention_mask,
    ...     pixel_values=inputs.pixel_values,
    ... )
   >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

   >>> # save and load from pretrained
   >>> model.save_pretrained("vit-bert")
   >>> model = FlaxVisionTextDualEncoderModel.from_pretrained("vit-bert")

   >>> # inference
   >>> outputs = model(**inputs)
   >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
   >>> probs = jax.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
    ```
"""

# 覆盖 FlaxVisionTextDualEncoderModel 的调用文档字符串
overwrite_call_docstring(
    FlaxVisionTextDualEncoderModel,
    VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING + VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING,
)

# 追加替换 FlaxVisionTextDualEncoderModel 的返回文档字符串
append_replace_return_docstrings(
    FlaxVisionTextDualEncoderModel, output_type=FlaxCLIPOutput, config_class=_CONFIG_FOR_DOC
)
```