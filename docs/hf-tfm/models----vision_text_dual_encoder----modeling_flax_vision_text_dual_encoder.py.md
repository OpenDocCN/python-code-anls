# `.\models\vision_text_dual_encoder\modeling_flax_vision_text_dual_encoder.py`

```
# coding=utf-8
# 版权所有 2021 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 均在“按原样”基础上分发，无论是明示的还是暗示的。
# 有关特定语言的权限，请参阅许可证。
""" Flax VisionTextDualEncoder model."""

# 从 typing 模块导入必要的类型
from typing import Optional, Tuple

# 导入 Flax 的 linen 模块作为 nn
import flax.linen as nn
# 导入 JAX 库，并将其别名为 jax
import jax
# 导入 JAX 的 numpy 模块，并将其别名为 jnp
import jax.numpy as jnp
# 导入 flax.core.frozen_dict 模块中的 FrozenDict、freeze 和 unfreeze 方法
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# 导入 flax.traverse_util 模块中的 flatten_dict 和 unflatten_dict 方法
from flax.traverse_util import flatten_dict, unflatten_dict

# 从模块中导入一些函数和类
from ...modeling_flax_utils import FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import add_start_docstrings, logging
# 从 auto 模块中导入 AutoConfig 类
from ..auto.configuration_auto import AutoConfig
# 从 auto 模块中导入 FLAX_MODEL_MAPPING 和 FlaxAutoModel 类
from ..auto.modeling_flax_auto import FLAX_MODEL_MAPPING, FlaxAutoModel
# 从 clip 模块中导入 FlaxCLIPOutput 和 FlaxCLIPVisionModel 类
from ..clip.modeling_flax_clip import FlaxCLIPOutput, FlaxCLIPVisionModel
# 从 configuration_vision_text_dual_encoder 模块中导入 VisionTextDualEncoderConfig 类
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置
_CONFIG_FOR_DOC = "VisionTextDualEncoderConfig"

# VisionTextDualEncoder 类的文档字符串，包含详细说明和用法示例
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
    # 导入必要的 JAX 模块，包括 JIT 编译、自动微分、向量化和并行化
    import jax
    import jax.numpy as jnp
    from flax.training.common import VisionTextDualEncoderConfig
    from transformers import FlaxPreTrainedModel
    
    # 函数定义：初始化模型配置
    def __init__(self, config: VisionTextDualEncoderConfig):
        # 模型配置，包含所有模型参数的类
        self.config = config
    
    # 函数定义：设置计算的数据类型
    def set_dtype(self, dtype=jnp.float32):
        """
        设置计算的数据类型。
    
        Parameters:
            dtype (jax.numpy.dtype, optional, default=jax.numpy.float32):
                计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）和 `jax.numpy.bfloat16`（在 TPU 上）。
                可以用于启用混合精度训练或在 GPU 或 TPU 上进行半精度推理。如果指定了 dtype，则所有的计算将使用给定的 dtype。
    
                **注意：这只指定了计算的数据类型，不影响模型参数的数据类型。**
    
                如果您希望更改模型参数的数据类型，请参见 `~FlaxPreTrainedModel.to_fp16` 和 `~FlaxPreTrainedModel.to_bf16`。
        """
        self.dtype = dtype
"""
定义了一个字符串常量 VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING，用于存储文档字符串。

Args:
    input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
        输入序列标记的索引，可以通过 AutoTokenizer 获取。默认情况下将忽略填充部分。
        
        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        遮罩，用于避免在填充标记索引上执行注意力。遮罩值选择在 `[0, 1]`：
        
        - 1 表示 **未遮罩** 的标记，
        - 0 表示 **已遮罩** 的标记。
        
        [什么是注意力遮罩？](../glossary#attention-mask)
    position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]`。
        
        [什么是位置 ID？](../glossary#position-ids)
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。默认情况下将忽略填充部分。可以使用图像处理器获取像素值（例如，如果使用 ViT 作为编码器，应使用 `AutoImageProcessor`）。
        
        [ViTImageProcessor.__call__] 获取更多细节。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。
    return_dict (`bool`, *optional*):
        是否返回 `~utils.ModelOutput` 而不是普通元组。
"""
class FlaxVisionTextDualEncoderModule(nn.Module):
    config: VisionTextDualEncoderConfig
    dtype: jnp.dtype = jnp.float32
    # 设置函数的初始化操作，准备模型和参数配置
    def setup(self):
        # 从配置对象中获取视觉模型和文本模型的配置信息
        vision_config = self.config.vision_config
        text_config = self.config.text_config

        # 设置视觉嵌入维度和文本嵌入维度
        self.vision_embed_dim = vision_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        self.projection_dim = self.config.projection_dim

        # 根据视觉模型和文本模型的配置选择相应的模型类
        vision_module = FLAX_MODEL_MAPPING.get(self.config.vision_config.__class__, FlaxCLIPVisionModel).module_class
        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class

        # 初始化视觉模型和文本模型
        self.vision_model = vision_module(vision_config, dtype=self.dtype)
        self.text_model = text_module(text_config, dtype=self.dtype)

        # 初始化视觉和文本的投影层
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

        # 初始化logit的缩放系数，并将其作为模型的参数
        self.logit_scale = self.param(
            "logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []
        )

    # 定义模型的调用方法，处理输入并返回模型的输出
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
            # 如果 return_dict 不为 None，则使用给定的 return_dict；否则使用对象自身的配置中的 return_dict
            return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 使用视觉模型处理像素值，获取视觉输出，包括注意力权重和隐藏状态
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 使用文本模型处理输入，获取文本输出，包括注意力权重和隐藏状态
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

        # 从视觉输出中获取图像嵌入，并通过投影层进行处理
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # 从文本输出中获取文本嵌入，并通过投影层进行处理
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # 对图像嵌入进行标准化处理
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        # 对文本嵌入进行标准化处理
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # 使用余弦相似度计算文本和图像嵌入之间的逻辑相似性得分
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        # 如果 return_dict 为 False，则返回包含多个输出的元组
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        # 如果 return_dict 为 True，则返回自定义的输出对象 FlaxCLIPOutput
        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
# 将文本和视觉输入编码成嵌入向量的模型，继承自FlaxPreTrainedModel
@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class FlaxVisionTextDualEncoderModel(FlaxPreTrainedModel):
    # 使用VisionTextDualEncoderConfig作为配置类
    config_class = VisionTextDualEncoderConfig
    # 使用FlaxVisionTextDualEncoderModule作为模块类
    module_class = FlaxVisionTextDualEncoderModule

    def __init__(
        self,
        config: VisionTextDualEncoderConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果不初始化，则抛出错误
        if not _do_init:
            raise ValueError(
                "`FlaxVisionTextDualEncoderModel` cannot be created without initializing, `_do_init` must be `True`."
            )

        # 如果未提供输入形状，则使用默认的输入形状
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))

        # 创建模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，传入配置、模块、输入形状、种子和数据类型
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        # 生成位置编码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        # 生成token类型编码，默认为1
        token_type_ids = jnp.ones_like(input_ids)
        # 生成注意力掩码，默认为全1
        attention_mask = jnp.ones_like(input_ids)

        # 生成像素值，使用正态分布随机数初始化
        pixel_values = jax.random.normal(rng, input_shape[1])

        # 分割随机数生成器，用于参数和dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块，获取随机参数
        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids, token_type_ids)[
            "params"
        ]

        # 如果提供了参数，则使用提供的参数替换缺失的随机参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

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
        #```
dropout_rng: jax.random.PRNGKey = None,
train: bool = False,
output_attentions: Optional[bool] = None,
output_hidden_states: Optional[bool] = None,
return_dict: Optional[bool] = None,
        ):
            # 如果 output_attentions 不为 None，则使用其值；否则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 不为 None，则使用其值；否则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 return_dict 不为 None，则使用其值；否则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 转置像素值数组，调整维度顺序
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

            # 如果 position_ids 为 None，则创建一个与 input_ids 最后一个维度广播兼容的数组
            if position_ids is None:
                position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

            # 如果 token_type_ids 为 None，则创建一个与 input_ids 形状相同的全零数组
            if token_type_ids is None:
                token_type_ids = jnp.zeros_like(input_ids)

            # 如果 attention_mask 为 None，则创建一个与 input_ids 形状相同的全一数组
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)

            # 处理任何需要的伪随机数发生器 PRNG
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            # 调用 self.module.apply 方法，传递相关参数进行模型应用
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

        # 定义一个方法 get_text_features，接受多个参数，用于获取文本特征
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
        # 如果未提供 position_ids 参数，则使用 input_ids 的长度广播生成位置 IDs
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供 token_type_ids 参数，则生成与 input_ids 形状相同的全零张量
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果未提供 attention_mask 参数，则生成与 input_ids 形状相同的全一张量
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理可能需要的任何伪随机数发生器（PRNG）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, token_type_ids, deterministic):
            # 调用文本模型获取文本输出
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                deterministic=deterministic,
            )
            # 从文本输出中获取汇聚输出
            pooled_output = text_outputs[1]
            # 应用文本投影层获得文本特征
            text_features = module.text_projection(pooled_output)
            return text_features

        # 调用模块的 apply 方法来应用参数和输入数据进行前向计算
        return self.module.apply(
            {"params": params or self.params},  # 提供模型参数
            jnp.array(input_ids, dtype="i4"),  # 输入的序列 token IDs
            jnp.array(attention_mask, dtype="i4"),  # 输入的注意力掩码
            jnp.array(position_ids, dtype="i4"),  # 输入的位置 IDs
            jnp.array(token_type_ids, dtype="i4"),  # 输入的 token 类型 IDs
            not train,  # 是否是推理模式
            method=_get_features,  # 调用的方法来获取特征
            rngs=rngs,  # 伪随机数发生器的字典
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

        # 定义一个内部函数，用于从视觉模型中提取特征
        def _get_features(module, pixel_values, deterministic):
            # 调用视觉模型，传入像素值和确定性参数，获取视觉模型的输出
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            # 提取汇总输出（通常是第二个输出）
            pooled_output = vision_outputs[1]  # pooled_output
            # 将汇总输出应用于视觉投影层，获取图像特征
            image_features = module.visual_projection(pooled_output)
            return image_features

        # 调用当前对象所包含的模块的 apply 方法，将参数和数据传入视觉模型处理函数
        return self.module.apply(
            {"params": params or self.params},  # 使用给定的参数或对象的参数
            jnp.array(pixel_values, dtype=jnp.float32),  # 将像素值转换为 jax 数组
            not train,  # 确定是否为训练模式
            method=_get_features,  # 指定处理方法为 _get_features 函数
            rngs=rngs,  # 传入任何可能需要的随机数生成器
        )

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
# 定义 VisionTextDualEncoderModel 的文档字符串，包含函数的返回值和示例用法
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

    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    >>> image_processor = AutoImageProcesor.from_pretrained("google/vit-base-patch16-224")
    >>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    >>> model = FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(
    ...     "google/vit-base-patch16-224", "google-bert/bert-base-uncased"
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

# 调用 overwrite_call_docstring 函数，用于替换 FlaxVisionTextDualEncoderModel 类的文档字符串
overwrite_call_docstring(
    FlaxVisionTextDualEncoderModel,
    VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING + VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING,
)

# 调用 append_replace_return_docstrings 函数，用于附加和替换 FlaxVisionTextDualEncoderModel 类的返回值文档字符串
append_replace_return_docstrings(
    FlaxVisionTextDualEncoderModel, output_type=FlaxCLIPOutput, config_class=_CONFIG_FOR_DOC
)
```