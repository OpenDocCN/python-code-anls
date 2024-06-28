# `.\models\clip\modeling_flax_clip.py`

```
# 导入所需模块和类
from typing import Any, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入模型输出相关的类
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
# 导入模型工具函数和类
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 导入通用工具函数和类
from ...utils import ModelOutput, add_start_docstrings, logging
# 导入相关配置类
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# CLIP模型的起始文档字符串，提供模型介绍和使用说明
CLIP_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    # 参数说明部分：config 是一个 CLIPConfig 类型的对象，包含模型的所有参数。
    # 通过传入一个配置文件初始化，不会加载模型的权重，只加载配置信息。
    # 查看 `FlaxPreTrainedModel.from_pretrained` 方法可以加载模型权重。
    # dtype 是计算数据的数据类型，默认为 `jax.numpy.float32`。
    # 可以选择 `jax.numpy.float32`, `jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）。
    # 这可以用于在GPU或TPU上启用混合精度训练或半精度推断。
    # 如果指定了dtype，则所有计算将使用给定的dtype执行。
    # 注意：这只指定计算的dtype，不影响模型参数的dtype。
    # 如果希望更改模型参数的dtype，请参阅 `FlaxPreTrainedModel.to_fp16` 和 `FlaxPreTrainedModel.to_bf16`。
"""
CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_INPUTS_DOCSTRING = r"""
    Placeholder for combining textual and visual inputs documentation for the CLIP model.
"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@flax.struct.dataclass
class FlaxCLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`jnp.ndarray` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: jnp.ndarray = None  # 文本嵌入，通过将投影层应用于[`FlaxCLIPTextModel`]的汇聚输出获得
    last_hidden_state: jnp.ndarray = None  # 模型最后一层的隐藏状态输出，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None  # 可选，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回，元组中包含每层输出的隐藏状态
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None  # 可选，当传递`output_attentions=True`或`config.output_attentions=True`时返回，元组中包含每层的注意力权重
"""


@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    """
    Args:
        logits_per_image: (`jnp.ndarray` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`jnp.ndarray` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds: (`jnp.ndarray` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        image_embeds: (`jnp.ndarray` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPVisionModel`].
        text_model_output: (`FlaxBaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPTextModel`].
        vision_model_output: (`FlaxBaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPVisionModel`].
    """

    logits_per_image: jnp.ndarray = None  # 图像与文本嵌入之间的标量乘积得分，形状为`(image_batch_size, text_batch_size)`，表示图像与文本之间的相似度分数
    logits_per_text: jnp.ndarray = None  # 文本与图像嵌入之间的标量乘积得分，形状为`(text_batch_size, image_batch_size)`，表示文本与图像之间的相似度分数
    text_embeds: jnp.ndarray = None  # 通过将投影层应用于[`FlaxCLIPTextModel`]的汇聚输出获得的文本嵌入
    image_embeds: jnp.ndarray = None  # 通过将投影层应用于[`FlaxCLIPVisionModel`]的汇聚输出获得的图像嵌入
    # 定义两个属性，分别用于存储文本模型和视觉模型的输出，初始值为None
    text_model_output: FlaxBaseModelOutputWithPooling = None
    vision_model_output: FlaxBaseModelOutputWithPooling = None
    
    # 定义一个方法，将对象的属性转换为元组返回
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 对于对象的每个属性，如果属性不是"text_model_output"或"vision_model_output"，直接取其值
            # 如果属性是"text_model_output"或"vision_model_output"，调用其to_tuple()方法进行转换
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()  # 遍历对象的所有属性名
        )
# 定义 FlaxCLIPVisionEmbeddings 类，继承自 nn.Module，用于视觉嵌入处理
class FlaxCLIPVisionEmbeddings(nn.Module):
    # 类属性 config 表示 CLIPVisionConfig 的配置
    config: CLIPVisionConfig
    # 类属性 dtype 表示数据类型，默认为 jnp.float32

    # 初始化方法 setup，用于设置模型结构和参数
    def setup(self):
        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = self.config.hidden_size
        # 从配置中获取图像大小和patch大小
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        # 初始化类别嵌入向量，命名为 class_embedding，使用正态分布初始化
        self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (embed_dim,))

        # 初始化 patch 嵌入层，使用卷积操作，无偏置，数据类型为 dtype
        self.patch_embedding = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )

        # 计算图像分割成 patch 后的总数
        self.num_patches = (image_size // patch_size) ** 2
        # 计算位置嵌入的总数，包括类别嵌入
        num_positions = self.num_patches + 1
        # 初始化位置嵌入层，使用正态分布初始化
        self.position_embedding = nn.Embed(num_positions, embed_dim, embedding_init=jax.nn.initializers.normal())
        # 初始化位置编号，用于确定每个位置的嵌入
        self.position_ids = jnp.expand_dims(jnp.arange(0, num_positions, dtype="i4"), axis=0)

    # 实现 __call__ 方法，用于执行模型的前向传播
    def __call__(self, pixel_values):
        # 对输入的像素值进行 patch 嵌入处理
        patch_embeds = self.patch_embedding(pixel_values)
        # 获取批量大小、高度、宽度和通道数
        batch_size, height, width, channels = patch_embeds.shape
        # 将 patch 嵌入重新形状为 (批量大小, 高度*宽度, 通道数)
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        # 扩展类别嵌入到每个图像片段，以便与 patch 嵌入连接
        class_embeds = jnp.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        # 将类别嵌入和 patch 嵌入连接起来形成最终的嵌入表示
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
        # 将位置嵌入加到最终嵌入表示中
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# 定义 FlaxCLIPTextEmbeddings 类，继承自 nn.Module，用于文本嵌入处理
class FlaxCLIPTextEmbeddings(nn.Module):
    # 类属性 config 表示 CLIPTextConfig 的配置
    config: CLIPTextConfig
    # 类属性 dtype 表示数据类型，默认为 jnp.float32

    # 初始化方法 setup，用于设置模型结构和参数
    def setup(self):
        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = self.config.hidden_size

        # 初始化 token 嵌入层，使用正态分布初始化
        self.token_embedding = nn.Embed(self.config.vocab_size, embed_dim, embedding_init=jax.nn.initializers.normal())
        # 初始化位置嵌入层，使用正态分布初始化
        self.position_embedding = nn.Embed(
            self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal()
        )
        # 初始化位置编号，用于确定每个位置的嵌入
        self.position_ids = jnp.expand_dims(
            jnp.arange(0, self.config.max_position_embeddings, dtype="i4"), axis=(0, 1)
        )

    # 实现 __call__ 方法，用于执行模型的前向传播
    def __call__(self, input_ids, position_ids):
        # 将输入的 token 编号转换为对应的 token 嵌入
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        # 获取对应位置编号的位置嵌入
        position_embeds = self.position_embedding(position_ids.astype("i4"))

        # 将 token 嵌入和位置嵌入相加得到最终的嵌入表示
        embeddings = input_embeds + position_embeds
        return embeddings


# 定义 FlaxCLIPAttention 类，继承自 nn.Module，用于注意力机制处理
class FlaxCLIPAttention(nn.Module):
    # 类属性 config 表示 CLIPTextConfig 或 CLIPVisionConfig 的配置
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    # 类属性 dtype 表示数据类型，默认为 jnp.float32
    # 设置函数，初始化模型的注意力相关参数
    def setup(self):
        # 设置嵌入维度为隐藏大小
        self.embed_dim = self.config.hidden_size
        # 设置注意力头的数量
        self.num_heads = self.config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查embed_dim是否能被num_heads整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 设置缩放因子
        self.scale = self.head_dim**-0.5
        # 设置注意力的dropout率
        self.dropout = self.config.attention_dropout

        # 初始化键、值、查询、输出的线性投影层
        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

        # 根据配置确定是否是有因果关系的注意力
        self.causal = isinstance(self.config, CLIPTextConfig)
        # 如果是因果关系注意力，则创建因果关系的掩码
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="i4"))

    # 将隐藏状态按照头的数量和头的维度进行分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割后的头重新合并成原始的维度
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 定义模型的调用方法，用于执行自注意力机制
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        ):
            # 使用 self.q_proj 对隐藏状态进行查询投影
            query = self.q_proj(hidden_states)
            # 使用 self.k_proj 对隐藏状态进行键投影
            key = self.k_proj(hidden_states)
            # 使用 self.v_proj 对隐藏状态进行值投影
            value = self.v_proj(hidden_states)

            # 将查询结果按头数分割
            query = self._split_heads(query)
            # 将键结果按头数分割
            key = self._split_heads(key)
            # 将值结果按头数分割
            value = self._split_heads(value)

            # 初始化因果注意力掩码
            causal_attention_mask = None
            if self.causal:
                # 如果开启因果模式，则根据查询和键的长度创建因果注意力掩码
                query_length, key_length = query.shape[1], key.shape[1]
                causal_attention_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

            # 整合外部传入的注意力掩码和因果注意力掩码
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_mask = combine_masks(attention_mask, causal_attention_mask, dtype="i4")
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask
            elif attention_mask is not None:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            # 根据最终得到的注意力掩码生成注意力偏置
            if attention_mask is not None:
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
                )
            else:
                attention_bias = None

            # 初始化 dropout 的随机数生成器
            dropout_rng = None
            if not deterministic and self.dropout > 0.0:
                dropout_rng = self.make_rng("dropout")

            # 计算注意力权重
            attn_weights = dot_product_attention_weights(
                query,
                key,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=None,
            )

            # 根据注意力权重计算注意力输出
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
            # 合并多头注意力的输出
            attn_output = self._merge_heads(attn_output)
            # 对注意力输出进行最终的投影
            attn_output = self.out_proj(attn_output)

            # 根据需求决定返回的输出内容
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs
# 定义一个使用 CLIPTextConfig 或 CLIPVisionConfig 类型配置的神经网络模块
class FlaxCLIPMLP(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 模块的配置属性，可以是文本或视觉配置类型
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32

    # 模块初始化设置方法
    def setup(self):
        # 根据配置中指定的激活函数选择对应的激活函数
        self.activation_fn = ACT2FN[self.config.hidden_act]
        # 第一个全连接层，输入大小为配置中的 intermediate_size，使用正态分布初始化权重
        self.fc1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        # 第二个全连接层，输入大小为配置中的 hidden_size，使用正态分布初始化权重
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

    # 模块调用方法
    def __call__(self, hidden_states):
        # 使用第一个全连接层进行前向传播
        hidden_states = self.fc1(hidden_states)
        # 使用选择的激活函数进行激活
        hidden_states = self.activation_fn(hidden_states)
        # 使用第二个全连接层进行前向传播
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 定义一个使用 CLIPTextConfig 或 CLIPVisionConfig 类型配置的编码器层模块
class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 模块的配置属性，可以是文本或视觉配置类型
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32

    # 模块初始化设置方法
    def setup(self):
        # 自注意力机制
        self.self_attn = FlaxCLIPAttention(self.config, dtype=self.dtype)
        # 第一层归一化层，使用配置中指定的 epsilon 进行归一化
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 多层感知机（MLP）模块
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)
        # 第二层归一化层，使用配置中指定的 epsilon 进行归一化
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 模块调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states  # 保存输入的残差连接

        # 对输入进行第一层归一化处理
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力机制进行注意力计算
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]  # 更新隐藏状态为注意力输出的第一个元素
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 更新残差连接

        # 对更新后的隐藏状态进行第二层归一化处理
        hidden_states = self.layer_norm2(hidden_states)
        # 使用多层感知机（MLP）模块进行前向传播
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 将输出封装成元组

        # 如果需要输出注意力信息，则添加到输出中
        if output_attentions:
            outputs += attn_outputs[1:]

        return outputs


# 定义一个使用 CLIPTextConfig 或 CLIPVisionConfig 类型配置的多层编码器层集合模块
class FlaxCLIPLayerCollection(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 模块的配置属性，可以是文本或视觉配置类型
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32

    # 模块初始化设置方法
    def setup(self):
        # 创建多层编码器层集合，每层使用 FlaxCLIPEncoderLayer 模块
        self.layers = [
            FlaxCLIPEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    # 模块调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,

        ):
            # 遍历每层编码器层进行处理
            for layer in self.layers:
                # 对隐藏状态进行编码器层处理
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                    output_attentions=output_attentions,
                )

            # 返回处理后的结果
            return hidden_states
    ):
        # 如果不输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果不输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历模型的每一层
        for layer in self.layers:
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播方法
            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions元组中
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到all_hidden_states元组中
            all_hidden_states += (hidden_states,)

        # 将最终的隐藏状态作为模型的输出
        outputs = (hidden_states,)

        if not return_dict:
            # 如果不返回字典形式的输出，则返回outputs中不为None的元素组成的元组
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput类的实例，其中包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxCLIPEncoder(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 定义config属性，可以是CLIPTextConfig或CLIPVisionConfig类型
    dtype: jnp.dtype = jnp.float32  # 定义dtype属性，默认为jnp.float32类型

    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, dtype=self.dtype)
        # 初始化layers属性为FlaxCLIPLayerCollection实例，使用给定的config和dtype参数

    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用self.layers对象，传递输入的嵌入向量inputs_embeds和其他可选参数，返回计算结果


class FlaxCLIPTextTransformer(nn.Module):
    config: CLIPTextConfig  # 定义config属性为CLIPTextConfig类型
    dtype: jnp.dtype = jnp.float32  # 定义dtype属性，默认为jnp.float32类型

    def setup(self):
        self.embeddings = FlaxCLIPTextEmbeddings(self.config, dtype=self.dtype)
        # 初始化embeddings属性为FlaxCLIPTextEmbeddings实例，使用给定的config和dtype参数
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        # 初始化encoder属性为FlaxCLIPEncoder实例，使用给定的config和dtype参数
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化final_layer_norm属性为nn.LayerNorm实例，使用给定的layer_norm_eps和dtype参数

        # For `pooled_output` computation
        self.eos_token_id = self.config.eos_token_id
        # 设置eos_token_id属性为config中的eos_token_id值

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 定义对象调用方法，接收输入参数，包括input_ids、attention_mask等
    # 如果没有指定output_attentions，则使用self.config.output_attentions
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # 如果没有指定output_hidden_states，则使用self.config.output_hidden_states
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # 如果没有指定return_dict，则使用self.config.use_return_dict
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # 使用input_ids和position_ids作为输入，生成hidden_states
    hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
    
    # 将hidden_states作为输入，并传入额外的参数，生成encoder_outputs
    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        deterministic=deterministic,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    # 选取encoder_outputs中的第一个元素作为last_hidden_state
    last_hidden_state = encoder_outputs[0]
    # 对last_hidden_state进行final_layer_norm处理
    last_hidden_state = self.final_layer_norm(last_hidden_state)
    
    # 如果eos_token_id等于2，则执行以下逻辑
    if self.eos_token_id == 2:
        # 从last_hidden_state中取出特定位置的特征，形成pooled_output
        pooled_output = last_hidden_state[jnp.arange(last_hidden_state.shape[0]), input_ids.argmax(axis=-1)]
    else:
        # 处理eos_token_id不等于2的情况
        pooled_output = last_hidden_state[
            jnp.arange(last_hidden_state.shape[0]), (input_ids == self.eos_token_id).argmax(axis=-1)
        ]
    
    # 如果return_dict为False，则返回last_hidden_state, pooled_output和encoder_outputs的其他部分
    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
    # 如果return_dict为True，则返回FlaxBaseModelOutputWithPooling对象
    return FlaxBaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
# 定义一个名为 FlaxCLIPVisionTransformer 的类，继承自 nn.Module
class FlaxCLIPVisionTransformer(nn.Module):
    # 类变量 config，指定为 CLIPVisionConfig 类型
    config: CLIPVisionConfig
    # 类变量 dtype，默认为 jnp.float32 类型
    dtype: jnp.dtype = jnp.float32

    # 初始化函数 setup，用于设置模型的组件
    def setup(self):
        # 创建 FlaxCLIPVisionEmbeddings 实例，并传入 config 和 dtype 参数
        self.embeddings = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)
        # 创建 nn.LayerNorm 实例，用于前层归一化，设定 epsilon 参数为 config 的 layer_norm_eps
        self.pre_layrnorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 FlaxCLIPEncoder 实例，并传入 config 和 dtype 参数
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        # 创建 nn.LayerNorm 实例，用于后层归一化，设定 epsilon 参数为 config 的 layer_norm_eps
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 定义调用函数，接受多个参数
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        # 根据参数设定 output_attentions，默认为 config 中的 output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数设定 output_hidden_states，默认为 config 中的 output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数设定 return_dict，默认为 config 中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 embeddings 对象处理输入的像素值，得到隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行前层归一化处理
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用 encoder 对象处理归一化后的隐藏状态
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器输出中提取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后一个隐藏状态进行池化操作，提取池化输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层归一化处理
        pooled_output = self.post_layernorm(pooled_output)

        # 如果 return_dict 为 False，则返回元组形式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 FlaxBaseModelOutputWithPooling 的实例
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 定义一个名为 FlaxCLIPTextPreTrainedModel 的类，继承自 FlaxPreTrainedModel
class FlaxCLIPTextPreTrainedModel(FlaxPreTrainedModel):
    # 类变量 config_class，指定为 CLIPTextConfig 类型
    config_class = CLIPTextConfig
    # 类变量 module_class，默认为 None
    module_class: nn.Module = None

    # 初始化函数，接受多个参数，包括一个 config 对象
    def __init__(
        self,
        config: CLIPTextConfig,
        input_shape=(1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据传入的 config 参数和其他参数创建 module 对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，初始化模型
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型权重的方法，使用给定的随机数生成器和输入形状，可选地使用现有的参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建位置编码张量，广播到输入形状的维度
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 创建注意力掩码张量，形状与输入张量相同，并初始化为全1
        attention_mask = jnp.ones_like(input_ids)

        # 分离随机数生成器为参数初始化和dropout层
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模型的初始化方法初始化随机的参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids)["params"]

        # 如果存在现有参数，则将随机生成的参数与现有参数进行合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将缺失的键从随机参数复制到现有参数
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结并返回合并后的参数字典
            return freeze(unflatten_dict(params))
        else:
            # 直接返回随机生成的参数字典
            return random_params

    # 模型对象的调用方法，接受一系列输入参数并返回模型的输出
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果未提供位置编码，则使用广播到输入张量形状的默认位置编码
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则创建一个与输入张量形状相同的全1掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理可能需要的任何随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 应用模型的前向传播方法并返回结果
        return self.module.apply(
            {"params": params or self.params},  # 模型参数，可以是传入的参数或者模型自身的参数
            jnp.array(input_ids, dtype="i4"),   # 输入张量，转换为32位整数
            jnp.array(attention_mask, dtype="i4"),  # 注意力掩码张量，转换为32位整数
            jnp.array(position_ids, dtype="i4"),    # 位置编码张量，转换为32位整数
            not train,          # 是否处于推理模式（训练模式取反）
            output_attentions,  # 是否输出注意力权重
            output_hidden_states,  # 是否输出隐藏状态
            return_dict,       # 是否以字典形式返回结果
            rngs=rngs,         # 随机数生成器字典
        )
# 定义一个继承自FlaxPreTrainedModel的新模型类，用于视觉任务的预训练模型
class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为CLIPVisionConfig
    config_class = CLIPVisionConfig
    # 主要输入的名称为"pixel_values"
    main_input_name = "pixel_values"
    # 模块类的类型暂未指定
    module_class: nn.Module = None

    # 初始化方法，接收多个参数包括config、input_shape等
    def __init__(
        self,
        config: CLIPVisionConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果未指定input_shape，默认为(1, config.image_size, config.image_size, 3)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        # 使用给定的config和dtype创建模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，传递config、module等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重的方法，接收随机数种子rng、输入形状input_shape、参数params等
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 使用正态分布生成输入张量pixel_values
        pixel_values = jax.random.normal(rng, input_shape)

        # 分割rng以获取参数rng和dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块的参数，返回随机生成的参数random_params
        random_params = self.module.init(rngs, pixel_values)["params"]

        # 如果提供了params，则将缺失的键从random_params复制到params中，并返回新的params
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 模型的调用方法，接收多个参数如pixel_values、params等
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果output_attentions、output_hidden_states或return_dict未指定，则使用config中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 将输入张量pixel_values转置为适合模块处理的形状
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 如果存在dropout_rng，则将其添加到rngs字典中
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的apply方法，传递params、pixel_values和其他参数，返回模型的输出
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


# 定义一个继承自FlaxPreTrainedModel的新模型类，用于通用的CLIP预训练模型
class FlaxCLIPPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为CLIPConfig
    config_class = CLIPConfig
    # 模块类的类型暂未指定
    module_class: nn.Module = None

    # 初始化方法，接收多个参数包括config、input_shape等
    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
    ):
        # 如果未提供输入形状，则使用默认形状：((1, 1), (1, vision_config.image_size, vision_config.image_size, 3))
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))
        
        # 根据指定的配置和参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        
        # 调用父类的初始化方法，传入配置、模块对象、输入形状等参数进行初始化
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全部置零
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        
        # 生成位置编码，广播到与输入张量相同的形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        
        # 创建注意力掩码，与输入张量形状相同，全部置一
        attention_mask = jnp.ones_like(input_ids)

        # 生成像素数值，服从正态分布
        pixel_values = jax.random.normal(rng, input_shape[1])

        # 划分随机数生成器为参数和丢弃的两部分
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化随机参数
        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids)["params"]

        if params is not None:
            # 如果提供了参数，则使用提供的参数，否则使用随机生成的参数
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
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果未提供位置编码，则根据输入张量的形状生成位置编码
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则生成与输入张量相同形状的全一掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 转置像素值，调整维度顺序
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 如果需要处理任何随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的应用方法，传入参数和数据，返回模块处理的结果
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        r"""
        Args:
            input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)

        Returns:
            text_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`FlaxCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        # 如果未提供位置 IDs，则创建一个广播以匹配输入 IDs 的长度
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力遮罩，则创建一个全1数组以表示全部注意
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果需要处理任何随机数生成器（PRNG）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义内部函数以获取文本特征
        def _get_features(module, input_ids, attention_mask, position_ids, deterministic):
            # 获取文本模型的输出
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )
            # 获取汇总后的输出
            pooled_output = text_outputs[1]
            # 应用文本投影层得到文本特征
            text_features = module.text_projection(pooled_output)
            return text_features

        # 应用模块的方法来获取文本特征
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    def get_image_features(
        self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train=False
    ):
    ):
        r"""
        Args:
            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        Returns:
            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`FlaxCLIPVisionModel`]

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # 转置像素值数组，调整通道顺序为(batch_size, height, width, num_channels)
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理可能需要的随机数发生器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            # 使用视觉模型处理像素值数组，获取视觉输出
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            # 提取池化后的输出
            pooled_output = vision_outputs[1]  # pooled_output
            # 将池化输出应用于视觉投影层，得到图像特征
            image_features = module.visual_projection(pooled_output)
            return image_features

        # 应用模块的特征提取方法，返回图像特征
        return self.module.apply(
            {"params": params or self.params},  # 模型参数
            jnp.array(pixel_values, dtype=jnp.float32),  # 转换后的像素值数组
            not train,  # 是否训练模式
            method=_get_features,  # 使用_get_features方法进行特征提取
            rngs=rngs,  # 随机数发生器字典
        )
class FlaxCLIPTextModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化文本模型为 FlaxCLIPTextTransformer，使用给定的配置和数据类型
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用文本模型的前向传播方法，传递给定的输入参数，并返回结果
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPTextModel(FlaxCLIPTextPreTrainedModel):
    # 模型类的模块类型设置为 FlaxCLIPTextModule
    module_class = FlaxCLIPTextModule


FLAX_CLIP_TEXT_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxCLIPTextModel

    >>> model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output  # pooled (EOS token) states
    ```
"""

# 覆盖 FlaxCLIPTextModel 类的 __call__ 方法的文档字符串，包括输入文档和模型输出示例
overwrite_call_docstring(FlaxCLIPTextModel, CLIP_TEXT_INPUTS_DOCSTRING + FLAX_CLIP_TEXT_MODEL_DOCSTRING)

# 追加或替换 FlaxCLIPTextModel 类的返回文档字符串，指定输出类型和配置类
append_replace_return_docstrings(
    FlaxCLIPTextModel, output_type=FlaxBaseModelOutputWithPooling, config_class=CLIPTextConfig
)


class FlaxCLIPTextModelWithProjectionModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化文本模型为 FlaxCLIPTextTransformer，使用给定的配置和数据类型
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)
        # 添加文本投影层，使用给定的投影维度和数据类型
        self.text_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 调用文本模型生成文本输出
            text_outputs = self.text_model(
                input_ids=input_ids,  # 输入的token IDs
                attention_mask=attention_mask,  # 注意力掩码
                position_ids=position_ids,  # 位置 IDs
                deterministic=deterministic,  # 是否确定性运行
                output_attentions=output_attentions,  # 是否输出注意力权重
                output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
                return_dict=return_dict,  # 是否返回字典形式的输出
            )

            # 从文本输出中获取汇聚的输出（一般是平均池化或CLS token的表示）
            pooled_output = text_outputs[1]
            # 将汇聚的输出通过文本投影层进行转换
            text_embeds = self.text_projection(pooled_output)

            # 如果不返回字典形式的输出，则返回元组
            if not return_dict:
                return (text_embeds, text_outputs[0]) + text_outputs[2:]

            # 如果返回字典形式的输出，则创建特定的输出对象
            return FlaxCLIPTextModelOutput(
                text_embeds=text_embeds,
                last_hidden_state=text_outputs.last_hidden_state,
                hidden_states=text_outputs.hidden_states,
                attentions=text_outputs.attentions,
            )
class FlaxCLIPTextModelWithProjection(FlaxCLIPTextPreTrainedModel):
    module_class = FlaxCLIPTextModelWithProjectionModule



# 定义一个类，继承自FlaxCLIPTextPreTrainedModel，用于文本模型与投影
class FlaxCLIPTextModelWithProjection(FlaxCLIPTextPreTrainedModel):
    # 指定模块类为FlaxCLIPTextModelWithProjectionModule
    module_class = FlaxCLIPTextModelWithProjectionModule


FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxCLIPTextModelWithProjection

    >>> model = FlaxCLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> text_embeds = outputs.text_embeds
    ```
"""

# 覆盖函数调用时的文档字符串，结合CLIP_TEXT_INPUTS_DOCSTRING和FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING
overwrite_call_docstring(
    FlaxCLIPTextModelWithProjection, CLIP_TEXT_INPUTS_DOCSTRING + FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING
)

# 追加或替换函数返回的文档字符串，输出类型为FlaxCLIPTextModelOutput，配置类为CLIPTextConfig
append_replace_return_docstrings(
    FlaxCLIPTextModelWithProjection, output_type=FlaxCLIPTextModelOutput, config_class=CLIPTextConfig
)


class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设置视觉模型为FlaxCLIPVisionTransformer，使用指定的配置和数据类型
        self.vision_model = FlaxCLIPVisionTransformer(self.config, dtype=self.dtype)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用视觉模型进行前向传播
        return self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPVisionModel(FlaxCLIPVisionPreTrainedModel):
    module_class = FlaxCLIPVisionModule




FLAX_CLIP_VISION_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, FlaxCLIPVisionModel

    >>> model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output  # pooled CLS states
    ```
"""

# 覆盖函数调用时的文档字符串，结合CLIP_VISION_INPUTS_DOCSTRING和FLAX_CLIP_VISION_MODEL_DOCSTRING
overwrite_call_docstring(FlaxCLIPVisionModel, CLIP_VISION_INPUTS_DOCSTRING + FLAX_CLIP_VISION_MODEL_DOCSTRING)

# 追加或替换函数返回的文档字符串，输出类型为FlaxBaseModelOutputWithPooling，配置类为CLIPVisionConfig
append_replace_return_docstrings(
    FlaxCLIPVisionModel, output_type=FlaxBaseModelOutputWithPooling, config_class=CLIPVisionConfig
)


class FlaxCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32
    # 设置模型的初始化过程
    text_config = self.config.text_config
    vision_config = self.config.vision_config

    # 设置投影维度和文本、视觉嵌入的维度
    self.projection_dim = self.config.projection_dim
    self.text_embed_dim = text_config.hidden_size
    self.vision_embed_dim = vision_config.hidden_size

    # 初始化文本模型和视觉模型，使用FlaxCLIPTextTransformer和FlaxCLIPVisionTransformer
    self.text_model = FlaxCLIPTextTransformer(text_config, dtype=self.dtype)
    self.vision_model = FlaxCLIPVisionTransformer(vision_config, dtype=self.dtype)

    # 初始化视觉投影层和文本投影层，设置投影维度和使用正态分布初始化权重，不使用偏置
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

    # 初始化logit_scale参数，设置为初始值为config.logit_scale_init_value的常数值
    self.logit_scale = self.param(
        "logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []
    )
        ):
        # 如果 return_dict 参数未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 调用视觉模型，传入像素值和其他相关参数，获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 调用文本模型，传入输入的 token IDs、注意力掩码、位置 IDs 等参数，获取文本模型的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉模型的输出中获取图像嵌入
        image_embeds = vision_outputs[1]
        # 通过视觉投影层处理图像嵌入
        image_embeds = self.visual_projection(image_embeds)

        # 从文本模型的输出中获取文本嵌入
        text_embeds = text_outputs[1]
        # 通过文本投影层处理文本嵌入
        text_embeds = self.text_projection(text_embeds)

        # 对图像嵌入进行归一化处理
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        # 对文本嵌入进行归一化处理
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # 计算余弦相似度作为 logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        # 如果不返回字典形式的结果，则按顺序返回元组
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        # 返回 FlaxCLIPOutput 对象，封装各类输出和模型状态
        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
# 使用装饰器为 FlaxCLIPModel 类添加起始文档字符串
@add_start_docstrings(CLIP_START_DOCSTRING)
# 将 FlaxCLIPPreTrainedModel 的模块类指定为 FlaxCLIPModule
class FlaxCLIPModel(FlaxCLIPPreTrainedModel):
    module_class = FlaxCLIPModule

# 定义 FLAX_CLIP_MODEL_DOCSTRING 常量，该常量包含关于 FlaxCLIPModel 类的详细文档字符串
FLAX_CLIP_MODEL_DOCSTRING = """
    Returns:
        描述函数返回的内容。

    Example:
        给出一个使用示例，展示模型如何使用。

    ```python
    >>> import jax
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, FlaxCLIPModel

    >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(
    ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True
    ... )

    >>> outputs = model(**inputs)
    >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    >>> probs = jax.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
    ```
"""

# 调用 overwrite_call_docstring 函数，将 CLIP_INPUTS_DOCSTRING 和 FLAX_CLIP_MODEL_DOCSTRING 合并作为 FlaxCLIPModel 类的文档字符串
overwrite_call_docstring(FlaxCLIPModel, CLIP_INPUTS_DOCSTRING + FLAX_CLIP_MODEL_DOCSTRING)

# 调用 append_replace_return_docstrings 函数，指定输出类型为 FlaxCLIPOutput，配置类为 CLIPConfig，为 FlaxCLIPModel 类附加和替换返回文档字符串
append_replace_return_docstrings(FlaxCLIPModel, output_type=FlaxCLIPOutput, config_class=CLIPConfig)
```