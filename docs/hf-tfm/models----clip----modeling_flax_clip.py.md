# `.\transformers\models\clip\modeling_flax_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，声明代码作者和许可证信息
# 该代码遵循 Apache License, Version 2.0 许可证
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 根据适用法律或书面同意，分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
# 导入所需的库和模块
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
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
# 获取日志记录器
logger = logging.get_logger(__name__)
# CLIP 模型的起始文档字符串
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
"""
    # 参数说明：
    # config: 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型相关的权重，只加载配置。查看`~FlaxPreTrainedModel.from_pretrained`方法以加载模型权重。
    # dtype: 计算的数据类型，默认为`jax.numpy.float32`。可以是`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和`jax.numpy.bfloat16`（在TPU上）之一。
    # 可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的dtype进行。
    # **请注意，这仅指定计算的dtype，不影响模型参数的dtype。**
    # 如果希望更改模型参数的dtype，请参阅`~FlaxPreTrainedModel.to_fp16`和`~FlaxPreTrainedModel.to_bf16`。
# CLIP 文本输入参数说明
CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下会忽略填充。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]`：

            - 1 表示**未被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# CLIP 视觉输入参数说明
CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下会忽略填充。像素值可以使用 [`AutoImageProcessor`] 获取。有关详细信息，请参阅
            [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# CLIP 输入参数说明
CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下会忽略填充。
            # 你可以使用 [`AutoTokenizer`] 来获取这些索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩以避免在填充的标记索引上执行注意力。遮罩值选择在 `[0, 1]` 之间：
            # - 1 表示**未屏蔽**的标记，
            # - 0 表示**已屏蔽**的标记。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
            # [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充。你可以使用 [`AutoImageProcessor`] 来获取像素值。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是纯元组。
```  
# 定义一个基于 Flax 的数据类，用于表示 CLIP 文本模型的输出结果
@flax.struct.dataclass
class FlaxCLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`jnp.ndarray` of shape `(batch_size, output_dim`):
            经过投影层处理后的文本嵌入向量，其维度为 `(batch_size, output_dim)`，表示文本的嵌入表示。
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列，其维度为 `(batch_size, sequence_length, hidden_size)`。

        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            如果设置了 `output_hidden_states=True` 或者 `config.output_hidden_states=True`，则返回隐藏状态的元组，包含了模型每一层的隐藏状态。
            每个元素都是 `jnp.ndarray`，维度为 `(batch_size, sequence_length, hidden_size)`。

        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            如果设置了 `output_attentions=True` 或者 `config.output_attentions=True`，则返回注意力权重的元组，包含了模型每一层的注意力权重。
            每个元素都是 `jnp.ndarray`，维度为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    text_embeds: jnp.ndarray = None
    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个基于 Flax 的数据类，用于表示 CLIP 的输出结果
@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    """
    Args:
        logits_per_image:(`jnp.ndarray` of shape `(image_batch_size, text_batch_size)`):
            图像嵌入向量与文本嵌入向量之间的缩放点积分数，表示图像与文本之间的相似度分数。
        logits_per_text:(`jnp.ndarray` of shape `(text_batch_size, image_batch_size)`):
            文本嵌入向量与图像嵌入向量之间的缩放点积分数，表示文本与图像之间的相似度分数。
        text_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
            经过投影层处理后的文本嵌入向量，其维度为 `(batch_size, output_dim)`，表示文本的嵌入表示。
        image_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
            经过投影层处理后的图像嵌入向量，其维度为 `(batch_size, output_dim)`，表示图像的嵌入表示。
        text_model_output(`FlaxBaseModelOutputWithPooling`):
            [`FlaxCLIPTextModel`] 的输出结果。
        vision_model_output(`FlaxBaseModelOutputWithPooling`):
            [`FlaxCLIPVisionModel`] 的输出结果。
    """

    logits_per_image: jnp.ndarray = None
    logits_per_text: jnp.ndarray = None
    text_embeds: jnp.ndarray = None
    image_embeds: jnp.ndarray = None
    # 初始化文本模型输出和视觉模型输出为 None
    text_model_output: FlaxBaseModelOutputWithPooling = None
    vision_model_output: FlaxBaseModelOutputWithPooling = None

    # 将对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        # 返回对象的属性组成的元组，如果属性名不是"text_model_output"或"vision_model_output"，则直接取属性值，
        # 否则调用对应属性的to_tuple()方法，并将结果添加到元组中
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class FlaxCLIPVisionEmbeddings(nn.Module):
    # 定义类，用于处理视觉嵌入
    config: CLIPVisionConfig
    # 指定配置参数类型为 CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32
    # 设置数据类型为 jnp.float32，默认为浮点数类型

    def setup(self):
        # 设置方法，用于初始化模型
        embed_dim = self.config.hidden_size
        # 获取隐藏尺寸
        image_size = self.config.image_size
        # 获取图像尺寸
        patch_size = self.config.patch_size
        # 获取补丁尺寸

        self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (embed_dim,))
        # 初始化类别嵌入参数，用于嵌入类别信息

        self.patch_embedding = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )
        # 定义卷积层，用于提取图像中的补丁特征，不使用偏置，使用指定的数据类型和初始化方式

        self.num_patches = (image_size // patch_size) ** 2
        # 计算图像中补丁的数量
        num_positions = self.num_patches + 1
        # 计算位置嵌入的数量
        self.position_embedding = nn.Embed(num_positions, embed_dim, embedding_init=jax.nn.initializers.normal())
        # 定义位置嵌入层，用于嵌入位置信息
        self.position_ids = jnp.expand_dims(jnp.arange(0, num_positions, dtype="i4"), axis=0)
        # 生成位置 ID，用于索引位置嵌入

    def __call__(self, pixel_values):
        # 定义调用方法，用于前向传播
        patch_embeds = self.patch_embedding(pixel_values)
        # 提取图像中的补丁特征
        batch_size, height, width, channels = patch_embeds.shape
        # 获取批量大小、高度、宽度和通道数
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))
        # 重新整形补丁特征

        class_embeds = jnp.expand_dims(self.class_embedding, axis=(0, 1))
        # 扩展类别嵌入的维度
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        # 复制类别嵌入以匹配批量大小
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
        # 将类别嵌入和补丁特征连接起来
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 添加位置嵌入
        return embeddings
        # 返回嵌入结果


class FlaxCLIPTextEmbeddings(nn.Module):
    # 定义类，用于处理文本嵌入
    config: CLIPTextConfig
    # 指定配置参数类型为 CLIPTextConfig
    dtype: jnp.dtype = jnp.float32
    # 设置数据类型为 jnp.float32，默认为浮点数类型

    def setup(self):
        # 设置方法，用于初始化模型
        embed_dim = self.config.hidden_size
        # 获取隐藏尺寸

        self.token_embedding = nn.Embed(self.config.vocab_size, embed_dim, embedding_init=jax.nn.initializers.normal())
        # 定义词嵌入层，用于嵌入输入的词汇信息
        self.position_embedding = nn.Embed(
            self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal()
        )
        # 定义位置嵌入层，用于嵌入位置信息
        self.position_ids = jnp.expand_dims(
            jnp.arange(0, self.config.max_position_embeddings, dtype="i4"), axis=(0, 1)
        )
        # 生成位置 ID，用于索引位置嵌入

    def __call__(self, input_ids, position_ids):
        # 定义调用方法，用于前向传播
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        # 对输入 ID 进行词嵌入
        position_embeds = self.position_embedding(position_ids.astype("i4"))
        # 对位置 ID 进行位置嵌入

        embeddings = input_embeds + position_embeds
        # 合并词嵌入和位置嵌入
        return embeddings
        # 返回嵌入结果


class FlaxCLIPAttention(nn.Module):
    # 定义类，用于处理注意力机制
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    # 指定配置参数类型为 CLIPTextConfig 或 CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32
    # 设置数据类型为 jnp.float32，默认为浮点数类型
    # 设置函数，初始化注意力头的维度和相关参数
    def setup(self):
        # 嵌入维度等于配置中的隐藏大小
        self.embed_dim = self.config.hidden_size
        # 注意力头的数量等于配置中的注意力头数量
        self.num_heads = self.config.num_attention_heads
        # 每个注意力头的维度等于嵌入维度除以注意力头数量
        self.head_dim = self.embed_dim // self.num_heads
        # 如果嵌入维度不能被注意力头数量整除，抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放参数等于每个注意力头维度的负平方根
        self.scale = self.head_dim**-0.5
        # 丢弃率等于配置中的注意力丢弃率
        self.dropout = self.config.attention_dropout

        # 初始化键、值和查询的线性投影层
        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

        # 检查是否是有因果关系的模型配置，如果是，则创建因果掩码
        self.causal = isinstance(self.config, CLIPTextConfig)
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="i4"))

    # 将隐藏状态分割成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将多个注意力头合并成原始隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 模型调用函数，接受隐藏状态、注意力掩码等输入，并输出注意力计算结果
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 使用查询投影层处理隐藏状态，得到查询向量
        query = self.q_proj(hidden_states)
        # 使用键投影层处理隐藏状态，得到键向量
        key = self.k_proj(hidden_states)
        # 使用值投影层处理隐藏状态，得到值向量
        value = self.v_proj(hidden_states)

        # 将查询向量进行头部分离操作
        query = self._split_heads(query)
        # 将键向量进行头部分离操作
        key = self._split_heads(key)
        # 将值向量进行头部分离操作
        value = self._split_heads(value)

        # 初始化因果注意力掩码为 None
        causal_attention_mask = None
        # 如果模型设置为因果，生成相应的因果注意力掩码
        if self.causal:
            query_length, key_length = query.shape[1], key.shape[1]
            causal_attention_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # 如果给定了外部注意力掩码并且存在因果注意力掩码，则将两者合并
        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = combine_masks(attention_mask, causal_attention_mask, dtype="i4")
        # 如果只有因果注意力掩码，则使用它
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        # 如果只有外部注意力掩码，则扩展维度以匹配张量形状
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # 根据注意力掩码生成注意力偏置
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # 如果模型非确定性且设置了 dropout，则生成 dropout 随机数生成器
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 使用点积注意力计算注意力权重
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

        # 使用注意力权重和值向量计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        # 合并注意力输出中的头部
        attn_output = self._merge_heads(attn_output)
        # 使用输出投影层处理合并后的注意力输出
        attn_output = self.out_proj(attn_output)

        # 如果需要输出注意力权重，则将注意力输出与权重一同返回，否则仅返回注意力输出
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
class FlaxCLIPMLP(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 定义配置参数类型
    dtype: jnp.dtype = jnp.float32  # 设置数据类型，默认为 jnp.float32

    def setup(self):
        self.activation_fn = ACT2FN[self.config.hidden_act]  # 获取激活函数
        self.fc1 = nn.Dense(  # 创建全连接层，用于 MLP 的第一层
            self.config.intermediate_size,  # 中间隐藏层的大小
            dtype=self.dtype,  # 数据类型
            kernel_init=jax.nn.initializers.normal(0.01),  # 初始化权重
        )
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))  # 创建全连接层，用于 MLP 的第二层

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)  # MLP 的第一层前向传播
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # MLP 的第二层前向传播
        return hidden_states


class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 定义配置参数类型
    dtype: jnp.dtype = jnp.float32  # 设置数据类型，默认为 jnp.float32

    def setup(self):
        self.self_attn = FlaxCLIPAttention(self.config, dtype=self.dtype)  # 创建自注意力层
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 创建 LayerNorm 层
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)  # 创建 MLP 层
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 创建另一个 LayerNorm 层

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states  # 记录残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 应用 LayerNorm 层
        attn_outputs = self.self_attn(  # 进行自注意力计算
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]  # 更新隐藏状态
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 记录残差连接
        hidden_states = self.layer_norm2(hidden_states)  # 应用另一个 LayerNorm 层
        hidden_states = self.mlp(hidden_states)  # 应用 MLP 层
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出结果

        if output_attentions:
            outputs += attn_outputs[1:]  # 输出注意力结果

        return outputs


class FlaxCLIPLayerCollection(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]  # 定义配置参数类型
    dtype: jnp.dtype = jnp.float32  # 设置数据类型，默认为 jnp.float32

    def setup(self):
        self.layers = [  # 创建多个 FlaxCLIPEncoderLayer 层
            FlaxCLIPEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ): 
        # 如果不输出注意力权重，则初始化一个空元组用于存储注意力权重
        all_attentions = () if output_attentions else None
        # 如果不输出隐藏状态，则初始化一个空元组用于存储隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历所有的层
        for layer in self.layers:
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播函数
            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
            # 获取当前层的输出隐藏状态
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构造模型的输出
        outputs = (hidden_states,)

        # 如果不返回字典，则返回模型的输出
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回字典形式的模型输出
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxCLIPEncoder(nn.Module):
    # FlaxCLIPEncoder 类定义，接受 CLIPTextConfig 或 CLIPVisionConfig 类型的配置
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在创建类实例时调用，初始化 FlaxCLIPLayerCollection 对象
    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, dtype=self.dtype)

    # 调用方法，用于前向传播
    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 FlaxCLIPLayerCollection 对象进行前向传播
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPTextTransformer(nn.Module):
    # FlaxCLIPTextTransformer 类定义，接受 CLIPTextConfig 类型的配置
    config: CLIPTextConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在创建类实例时调用，初始化 FlaxCLIPTextEmbeddings、FlaxCLIPEncoder 和 nn.LayerNorm 对象
    def setup(self):
        # 初始化 FlaxCLIPTextEmbeddings 对象
        self.embeddings = FlaxCLIPTextEmbeddings(self.config, dtype=self.dtype)
        # 初始化 FlaxCLIPEncoder 对象
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        # 初始化 nn.LayerNorm 对象，用于最终层归一化
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        # 用于计算 pooled_output 的结束标记 ID
        self.eos_token_id = self.config.eos_token_id

    # 调用方法，用于前向传播
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
        # 如果输出注意力权重参数不为 None，则使用参数值，否则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态参数不为 None，则使用参数值，否则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典参数不为 None，则使用参数值，否则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入的 token IDs 和位置 IDs 传递给嵌入层进行编码
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 将编码后的隐藏状态传递给编码器进行处理
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器输出中提取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后一个隐藏状态进行最终的层归一化处理
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 如果结束标记的 token ID 为 2
        if self.eos_token_id == 2:
            # `eos_token_id` 在 PR #24773 之前是错误的: 让我们保持这里所做的更改。
            # 具有这种 `eos_token_id` 的 CLIP 模型在配置中无法正确处理额外的新标记
            # ------------------------------------------------------------
            # 提取序列中每个序列的 EOS 标记的特征（eos_token_id 是每个序列中的最大值）
            pooled_output = last_hidden_state[jnp.arange(last_hidden_state.shape[0]), input_ids.argmax(axis=-1)]
        else:
            # 从最后一个隐藏状态中提取池化输出，根据结束标记的 token ID
            pooled_output = last_hidden_state[
                jnp.arange(last_hidden_state.shape[0]), (input_ids == self.eos_token_id).argmax(axis=-1)
            ]

        # 如果不需要返回字典，则返回元组形式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典，则返回带池化的 FlaxBaseModelOutputWithPooling 类型的结果
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class FlaxCLIPVisionTransformer(nn.Module):
    # 类型提示，指定 config 属性为 CLIPVisionConfig 类型
    config: CLIPVisionConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模型初始化方法
    def setup(self):
        # 初始化嵌入层
        self.embeddings = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)
        # 初始化前层归一化
        self.pre_layrnorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化编码器
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        # 初始化后层归一化
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 模型调用方法
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        # 若未指定输出注意力信息，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 若未指定输出隐藏状态信息，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 若未指定返回字典类型，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取嵌入层的隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行前层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 调用编码器获取编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 池化最后一个隐藏状态
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化后的输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 若不返回字典类型，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带有池化输出的 FlaxBaseModelOutputWithPooling 对象
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FlaxCLIPTextPreTrainedModel(FlaxPreTrainedModel):
    # 类型提示，指定配置类为 CLIPTextConfig
    config_class = CLIPTextConfig
    # 模块类未定义
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: CLIPTextConfig,
        input_shape=(1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 实例化模块类
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化权重方法，用于初始化模型的参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 根据输入张量的形状广播创建位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 创建全 1 的注意力掩码张量
        attention_mask = jnp.ones_like(input_ids)

        # 分割随机数生成器为参数随机数和 dropout 随机数
        params_rng, dropout_rng = jax.random.split(rng)
        # 构建随机数字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化模型参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids)["params"]

        # 如果已有参数，则使用已有参数初始化模型参数，否则返回随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 模型调用方法
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
        # 确定是否输出注意力矩阵，默认根据模型配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏层状态，默认根据模型配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典格式，默认根据模型配置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供位置编码，则根据输入张量的形状广播创建位置编码张量
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则创建全 1 的注意力掩码张量
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理任何需要的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 应用模块的前向传播方法
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
# 定义一个基于 Flax 的 CLIP 视觉预训练模型类，继承自 FlaxPreTrainedModel
class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为 CLIPVisionConfig
    config_class = CLIPVisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 模块类属性为 None
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: CLIPVisionConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果未提供输入形状，则默认为 (1, image_size, image_size, 3)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        # 使用给定的配置和参数初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 用正态分布随机初始化像素值张量
        pixel_values = jax.random.normal(rng, input_shape)

        # 分裂 PRNG
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的 init 方法初始化参数
        random_params = self.module.init(rngs, pixel_values)["params"]

        # 如果提供了初始参数，则将缺失的键用随机参数填充
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 模型调用方法
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
        # 确定是否输出注意力、隐藏状态、字典等
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 转置像素值张量的维度
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的 apply 方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


# 定义一个基于 Flax 的 CLIP 预训练模型类，继承自 FlaxPreTrainedModel
class FlaxCLIPPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为 CLIPConfig
    config_class = CLIPConfig
    # 模块类属性为 None
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果输入形状未指定，则设置默认形状
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))
        # 使用给定的配置和关键字参数实例化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的构造函数初始化对象
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        # 初始化位置编码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        # 初始化注意力掩码
        attention_mask = jnp.ones_like(input_ids)

        # 生成随机像素值
        pixel_values = jax.random.normal(rng, input_shape[1])

        # 分割 PRNGKey 以用于参数初始化和丢弃
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用初始化的参数和随机输入来初始化模块参数
        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids)["params"]

        # 如果提供了参数，则与随机参数进行合并
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
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 确定是否返回注意力矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否返回隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供位置编码，则根据输入张量形状生成位置编码
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则初始化全 1 的注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 转置像素值张量以匹配模块期望的形状
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理任何需要的 PRNGKey
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的 apply 方法来执行前向传播
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
    # 获取文本特征的方法，输入为模型输入的 token IDs，返回文本嵌入
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

        ```py
        >>> from transformers import AutoTokenizer, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 如果未提供位置 ID，则创建默认位置 ID，用于生成位置嵌入
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则创建全 1 的注意力掩码，表示不对任何 token 进行屏蔽
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理任何需要的 PRNG（伪随机数生成器）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义内部函数，用于获取文本特征
        def _get_features(module, input_ids, attention_mask, position_ids, deterministic):
            # 调用文本模型，获取文本输出
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )
            # 从文本输出中获取汇总后的输出，即汇总特征
            pooled_output = text_outputs[1]
            # 对汇总后的输出应用文本投影层，得到文本特征
            text_features = module.text_projection(pooled_output)
            return text_features

        # 应用模型并返回文本特征
        return self.module.apply(
            {"params": params or self.params},  # 模型参数
            jnp.array(input_ids, dtype="i4"),    # 输入 token IDs
            jnp.array(attention_mask, dtype="i4"),  # 注意力掩码
            jnp.array(position_ids, dtype="i4"),    # 位置 IDs
            not train,  # 是否为训练模式
            method=_get_features,   # 使用定义的内部函数获取特征
            rngs=rngs,   # 伪随机数生成器
        )

    # 获取图像特征的方法，输入为图像像素值，返回图像嵌入
    def get_image_features(
        self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train=False
    ):
        r"""
        Args:
            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
                像素值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 获得像素值。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。

        Returns:
            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): 应用投影层到 [`FlaxCLIPVisionModel`] 汇总输出得到的图像嵌入。

        Examples:

        ```py
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
        # 将像素值的通道维度移动到最后一个维度
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 如果需要，处理任何 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            # 获取视觉模型的输出
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            # 提取汇总输出（pooled_output）
            pooled_output = vision_outputs[1]
            # 应用视觉投影层得到图像特征
            image_features = module.visual_projection(pooled_output)
            return image_features

        # 应用模型并返回图像特征
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )
# 定义一个名为FlaxCLIPTextModule的类，继承自nn.Module
class FlaxCLIPTextModule(nn.Module):
    # 类属性，指定配置类型为CLIPTextConfig
    config: CLIPTextConfig
    # 类属性，指定数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，用于设置模型
    def setup(self):
        # 初始化文本模型，使用FlaxCLIPTextTransformer类
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)

    # 定义调用方法，用于执行模型的前向传播
    def __call__(
        self,
        input_ids,  # 输入的token ID
        attention_mask,  # 注意力掩码
        position_ids,  # 位置 ID
        deterministic: bool = True,  # 是否使用确定性推断
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否返回字典形式的输出
    ):
        # 调用文本模型的前向传播方法，传入参数并返回结果
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 定义一个名为FlaxCLIPTextModel的类，继承自FlaxCLIPTextPreTrainedModel类
class FlaxCLIPTextModel(FlaxCLIPTextPreTrainedModel):
    # 类属性，指定模块类为FlaxCLIPTextModule
    module_class = FlaxCLIPTextModule


# 用于文档字符串的字符串常量，描述了模型的返回值和示例用法
FLAX_CLIP_TEXT_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```py
    >>> from transformers import AutoTokenizer, FlaxCLIPTextModel

    >>> model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output  # pooled (EOS token) states
    ```
"""

# 更新FlaxCLIPTextModel的调用方法文档字符串
overwrite_call_docstring(FlaxCLIPTextModel, CLIP_TEXT_INPUTS_DOCSTRING + FLAX_CLIP_TEXT_MODEL_DOCSTRING)
# 追加或替换FlaxCLIPTextModel的返回值文档字符串
append_replace_return_docstrings(
    FlaxCLIPTextModel, output_type=FlaxBaseModelOutputWithPooling, config_class=CLIPTextConfig
)


# 定义一个名为FlaxCLIPTextModelWithProjectionModule的类，继承自nn.Module
class FlaxCLIPTextModelWithProjectionModule(nn.Module):
    # 类属性，指定配置类型为CLIPTextConfig
    config: CLIPTextConfig
    # 类属性，指定数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，用于设置模型
    def setup(self):
        # 初始化文本模型，使用FlaxCLIPTextTransformer类
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)
        # 初始化文本投影层，使用nn.Dense类
        self.text_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype)

    # 定义调用方法，用于执行模型的前向传播
    def __call__(
        self,
        input_ids,  # 输入的token ID
        attention_mask,  # 注意力掩码
        position_ids,  # 位置 ID
        deterministic: bool = True,  # 是否使用确定性推断
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否返回字典形式的输出
    ):  # 定义一个方法，接收一些参数并返回处理后的结果
        # 使用文本模型处理输入数据，返回文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,  # 输入的标识符
            attention_mask=attention_mask,  # 注意力掩码
            position_ids=position_ids,  # 位置标识符
            deterministic=deterministic,  # 是否确定性
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )

        # 从文本输出中提取池化的输出
        pooled_output = text_outputs[1]
        # 对池化的输出进行文本投影
        text_embeds = self.text_projection(pooled_output)

        # 如果不返回字典，则返回一个元组
        if not return_dict:
            return (text_embeds, text_outputs[0]) + text_outputs[2:]

        # 如果返回字典，则返回一个 FlaxCLIPTextModelOutput 对象
        return FlaxCLIPTextModelOutput(
            text_embeds=text_embeds,  # 文本嵌入
            last_hidden_state=text_outputs.last_hidden_state,  # 最后一个隐藏状态
            hidden_states=text_outputs.hidden_states,  # 隐藏状态
            attentions=text_outputs.attentions,  # 注意力权重
        )
class FlaxCLIPTextModelWithProjection(FlaxCLIPTextPreTrainedModel):
    # 使用自定义的模块类 FlaxCLIPTextModelWithProjectionModule
    module_class = FlaxCLIPTextModelWithProjectionModule


# FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING 是关于 FlaxCLIPTextModelWithProjection 的文档字符串
FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING = """
    Returns:

    Example:

    ```py
    >>> from transformers import AutoTokenizer, FlaxCLIPTextModelWithProjection

    >>> model = FlaxCLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> text_embeds = outputs.text_embeds
    ```
"""

# 用 FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING 覆盖 FlaxCLIPTextModelWithProjection 的调用文档字符串
overwrite_call_docstring(
    FlaxCLIPTextModelWithProjection, CLIP_TEXT_INPUTS_DOCSTRING + FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING
)

# 用 FlaxCLIPTextModelOutput 替换 FlaxCLIPTextModelWithProjection 的返回值文档字符串
# 使用 CLIPTextConfig 作为配置类
append_replace_return_docstrings(
    FlaxCLIPTextModelWithProjection, output_type=FlaxCLIPTextModelOutput, config_class=CLIPTextConfig
)


class FlaxCLIPVisionModule(nn.Module):
    # 用于处理 CLIPVisionConfig 的模块类
    config: CLIPVisionConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建 FlaxCLIPVisionTransformer 模型
        self.vision_model = FlaxCLIPVisionTransformer(self.config, dtype=self.dtype)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 vision_model 的 __call__ 方法
        return self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPVisionModel(FlaxCLIPVisionPreTrainedModel):
    # 使用 FlaxCLIPVisionModule 作为模块类
    module_class = FlaxCLIPVisionModule


# FLAX_CLIP_VISION_MODEL_DOCSTRING 是关于 FlaxCLIPVisionModel 的文档字符串
FLAX_CLIP_VISION_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```py
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

# 用 FLAX_CLIP_VISION_MODEL_DOCSTRING 覆盖 FlaxCLIPVisionModel 的调用文档字符串
overwrite_call_docstring(FlaxCLIPVisionModel, CLIP_VISION_INPUTS_DOCSTRING + FLAX_CLIP_VISION_MODEL_DOCSTRING)

# 用 FlaxBaseModelOutputWithPooling 替换 FlaxCLIPVisionModel 的返回值文档字符串
# 使用 CLIPVisionConfig 作为配置类
append_replace_return_docstrings(
    FlaxCLIPVisionModel, output_type=FlaxBaseModelOutputWithPooling, config_class=CLIPVisionConfig
)


class FlaxCLIPModule(nn.Module):
    # 处理 CLIPConfig 的模块类
    config: CLIPConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 设置模型参数和配置
    def setup(self):
        # 获取文本和视觉配置
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        # 设置投影维度
        self.projection_dim = self.config.projection_dim
        # 设置文本嵌入维度
        self.text_embed_dim = text_config.hidden_size
        # 设置视觉嵌入维度
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = FlaxCLIPTextTransformer(text_config, dtype=self.dtype)
        self.vision_model = FlaxCLIPVisionTransformer(vision_config, dtype=self.dtype)

        # 初始化视觉投影层
        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        # 初始化文本投影层
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

    # 模型调用函数
    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        # 如果 return_dict 为 None，则使用 self.config.return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 使用 vision_model 处理图像数据
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 使用 text_model 处理文本数据
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取图像的嵌入向量
        image_embeds = vision_outputs[1]
        # 对图像的嵌入向量进行投影
        image_embeds = self.visual_projection(image_embeds)

        # 获取文本的嵌入向量
        text_embeds = text_outputs[1]
        # 对文本的嵌入向量进行投影
        text_embeds = self.text_projection(text_embeds)

        # 对图像的嵌入向量进行归一化处理
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        # 对文本的嵌入向量进行归一化处理
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # 计算余弦相似度作为 logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        # 如果 return_dict 为 False，则返回一组元组
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        # 返回 FlaxCLIPOutput 对象
        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
# 将指定文档字符串添加到FlaxCLIPModel类中
@add_start_docstrings(CLIP_START_DOCSTRING)
class FlaxCLIPModel(FlaxCLIPPreTrainedModel):
    # 模块类设置为FlaxCLIPModule
    module_class = FlaxCLIPModule


# 定义FlaxCLIPModel的文档字符串
FLAX_CLIP_MODEL_DOCSTRING = """
    Returns:  # 返回说明

    Example:  # 示例

    ```py
    >>> import jax
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, FlaxCLIPModel

    >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # 从预训练模型中加载FlaxCLIPModel实例
    >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")  # 从预训练模型中加载AutoProcessor实例

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图片URL
    >>> image = Image.open(requests.get(url, stream=True).raw)  # 从URL获取图片并打开

    >>> inputs = processor(  # 使用processor处理输入，将文本和图像转换为模型的输入
    ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True
    ... )

    >>> outputs = model(**inputs)  # 使用模型对输入进行推断
    >>> logits_per_image = outputs.logits_per_image  # 这是图像文本相似度分数
    >>> probs = jax.nn.softmax(logits_per_image, axis=1)  # 我们可以对结果进行softmax操作以获得标签概率
    ```
"""

# 将指定的文档字符串覆盖FlaxCLIPModel类的调用文档字符串
overwrite_call_docstring(FlaxCLIPModel, CLIP_INPUTS_DOCSTRING + FLAX_CLIP_MODEL_DOCSTRING)

# 追加和替换FlaxCLIPModel类的返回文档字符串
append_replace_return_docstrings(FlaxCLIPModel, output_type=FlaxCLIPOutput, config_class=CLIPConfig)
```py  
```