# `.\transformers\models\beit\modeling_flax_beit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库和模块
from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxSequenceClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig

# 定义 FlaxBeitModelOutputWithPooling 类，继承自 FlaxBaseModelOutputWithPooling
@flax.struct.dataclass
class FlaxBeitModelOutputWithPooling(FlaxBaseModelOutputWithPooling):
    """
    Class for outputs of [`FlaxBeitModel`].

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

# BEIT_START_DOCSTRING 用于后续补充文档字符串
BEIT_START_DOCSTRING = r"""
    # 这个模型继承自 [`FlaxPreTrainedModel`]。查看超类文档以了解库实现的通用方法，比如下载、保存和从PyTorch模型转换权重等。
    
    # 这个模型还是一个 [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) 子类。将其用作常规的Flax linen Module，并参考Flax文档了解所有与一般用法和行为相关的事项。
    
    # 最后，这个模型支持内置的JAX功能，如:
    
    # - [即时 (JIT) 编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    # - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    # - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    # - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    
    # 参数:
    #     config ([`BeitConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型相关的权重，只会加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    #     dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    #         计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` (在GPU上) 和 `jax.numpy.bfloat16` (在TPU上) 中的一个。
    
    #         这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算都将使用给定的 `dtype` 进行。
    
    #         **请注意，这只指定了计算的dtype，不会影响模型参数的dtype。**
    
    #         如果要更改模型参数的dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            像素值。像素值可以使用 [`AutoImageProcessor`] 获得。有关详细信息，请参阅 [`AutoImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
"""


def relative_position_index_init(window_size: Tuple[int, int]) -> jnp.ndarray:
    """
    初始化每个窗口内每个标记的成对相对位置索引
    """
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # 从0开始偏移
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    relative_position_index = np.zeros(shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return jnp.array(relative_position_index)


def ones_with_scale(key, shape, scale, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * scale


class FlaxBeitDropPath(nn.Module):
    """每个样本的 DropPath（随机深度），当应用于残差块的主路径时。"""

    rate: float

    @nn.module.compact
    def __call__(self, inputs, deterministic: Optional[bool] = True):
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # 适用于不同维度张量，而不仅仅是二维卷积网络
            rng = self.make_rng("droppath")
            random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)
            binary_tensor = jnp.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output
```  
class FlaxBeitPatchEmbeddings(nn.Module):
    # 用于嵌入图像的类，继承自 nn.Module
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为 jnp.float32

    def setup(self):
        # 初始化函数，设置一些必要的参数
        self.num_channels = self.config.num_channels
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        patch_shape = (image_size // patch_size, image_size // patch_size)
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        # 使用卷积操作进行投影，将图像块映射到隐藏空间
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            # 使用正态分布初始化卷积核
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, pixel_values):
        # 调用函数，将像素值映射到隐藏空间
        num_channels = pixel_values.shape[-1]
        # 检查像素值的通道维度是否与配置中设置的通道数一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 使用卷积投影将像素值映射到隐藏空间
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        # 重新组织嵌入张量的形状，以适应后续处理
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxBeitEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为 jnp.float32

    def setup(self):
        # 初始化函数，设置一些必要的参数
        self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
        # 如果配置中启用了 mask token，则初始化 mask token
        if self.config.use_mask_token:
            self.mask_token = self.param("mask_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
        # 初始化图像块嵌入
        self.patch_embeddings = FlaxBeitPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        # 如果配置中启用了绝对位置嵌入，则初始化位置嵌入
        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.param(
                "position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.config.hidden_size)
            )
        # 初始化 dropout 层，用于随机丢弃一部分神经元，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    # 定义类的调用方法，用于处理输入的像素值
    def __call__(self, pixel_values, bool_masked_pos=None, deterministic=True):
        # 将像素值转换为嵌入表示
        embeddings = self.patch_embeddings(pixel_values)
        # 获取批次大小、序列长度和嵌入维度
        batch_size, seq_len, _ = embeddings.shape

        # 将类别标记进行广播以匹配嵌入维度，并转换为与嵌入相同的数据类型
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        cls_tokens = cls_tokens.astype(embeddings.dtype)

        # 如果存在被遮蔽的位置信息
        if bool_masked_pos is not None:
            # 将遮蔽标记进行广播以匹配嵌入维度，并转换为与嵌入相同的数据类型
            mask_tokens = jnp.broadcast_to(self.mask_token, (batch_size, seq_len, self.config.hidden_size))
            mask_tokens = mask_tokens.astype(embeddings.dtype)
            # 将被遮蔽的视觉标记替换为遮蔽标记
            w = jnp.expand_dims(bool_masked_pos, axis=-1)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 将类别标记与嵌入拼接在一起
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        # 如果使用绝对位置嵌入
        if self.config.use_absolute_position_embeddings:
            # 将绝对位置嵌入添加到嵌入中
            embeddings = embeddings + self.position_embeddings.astype(embeddings.dtype)

        # 应用 dropout，确定性参数用于控制是否采用确定性 dropout
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        # 返回嵌入
        return embeddings
# 定义一个名为FlaxBeitRelativePositionBias的类，继承自nn.Module
class FlaxBeitRelativePositionBias(nn.Module):
    # 类属性：Beit模型的配置信息
    config: BeitConfig
    # 类属性：窗口大小的元组，表示注意力机制的窗口大小
    window_size: Tuple[int, int]
    # 类属性：计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 计算相对位置偏置表中的相对距离数量
        num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        # 初始化相对位置偏置表参数，尺寸为(num_relative_distance, num_attention_heads)
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            (num_relative_distance, self.config.num_attention_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH
        # 初始化相对位置索引表
        self.relative_position_index = relative_position_index_init(self.window_size)

    # 调用方法
    def __call__(self):
        # 将相对位置索引表展平为一维
        index = self.relative_position_index.reshape(-1)
        # 定义相对位置偏置的形状
        shape = (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        # 根据索引从相对位置偏置表中获取相对位置偏置，并按指定形状重新组织
        relative_position_bias = self.relative_position_bias_table[index].reshape(shape)  # Wh*Ww,Wh*Ww,nH
        # 返回相对位置偏置，将通道维置于首位
        return jnp.transpose(relative_position_bias, (2, 0, 1))


# 定义一个名为FlaxBeitSelfAttention的类，继承自nn.Module
class FlaxBeitSelfAttention(nn.Module):
    # 类属性：Beit模型的配置信息
    config: BeitConfig
    # 类属性：窗口大小的元组，表示注意力机制的窗口大小
    window_size: Tuple[int, int]
    # 类属性：计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 检查隐藏大小是否是注意力头数的倍数，如果不是则抛出异常
        if self.config.hidden_size % self.config.num_attention_heads != 0 and not hasattr(
            self.config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {self.config.hidden_size,} is not a multiple of the number of attention "
                f"heads {self.config.num_attention_heads}."
            )

        # 初始化查询（query）、键（key）和值（value）的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化相对位置偏置对象，如果存在窗口大小则创建，否则为None
        self.relative_position_bias = (
            FlaxBeitRelativePositionBias(self.config, window_size=self.window_size, dtype=self.dtype)
            if self.window_size
            else None
        )

    # 调用方法
    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 使用查询（query）网络计算查询状态，并重塑形状以便进行注意力计算
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用值（value）网络计算值状态，并重塑形状以便进行注意力计算
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用键（key）网络计算键状态，并重塑形状以便进行注意力计算
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 初始化一个随机数生成器用于 dropout
        dropout_rng = None
        # 如果非确定性计算且设置了注意力 dropout 概率，则创建一个随机数生成器
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 初始化注意力偏置
        attention_bias = jnp.array(0.0, dtype=self.dtype)
        # 如果存在相对位置偏置，则添加到注意力偏置中
        if self.relative_position_bias is not None:
            attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
            attention_bias = attention_bias.astype(query_states.dtype)

        # 如果提供了共享的相对位置偏置，则将其添加到注意力偏置中
        if relative_position_bias is not None:
            attention_bias = attention_bias + relative_position_bias.astype(attention_bias.dtype)

        # 计算点积注意力权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 使用注意力权重和值状态计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 如果需要输出注意力权重，则返回注意力输出和注意力权重，否则仅返回注意力输出
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
class FlaxBeitSelfOutput(nn.Module):
    config: BeitConfig  # 配置参数对象
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个Dropout层，用于随机置零输入张量的部分元素，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        # 将隐藏状态通过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBeitAttention(nn.Module):
    config: BeitConfig  # 配置参数对象
    window_size: Tuple[int, int]  # 窗口大小
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个Beit自注意力层
        self.attention = FlaxBeitSelfAttention(self.config, self.window_size, dtype=self.dtype)
        # 创建一个Beit自注意力层的输出层
        self.output = FlaxBeitSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic=True, output_attentions: bool = False
    ):
        # 经过自注意力层
        attn_outputs = self.attention(
            hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        # 经过自注意力层的输出层
        attn_output = self.output(attn_output, deterministic=deterministic)

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxBeitIntermediate(nn.Module):
    config: BeitConfig  # 配置参数对象
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，用于Beit中间层的变换
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 选择激活函数
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        return hidden_states


class FlaxBeitOutput(nn.Module):
    config: BeitConfig  # 配置参数对象
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，用于Beit输出层的变换
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个Dropout层，用于随机置零输入张量的部分元素，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxBeitLayer(nn.Module):
    config: BeitConfig  # 配置参数对象
    window_size: Tuple[int, int]  # 窗口大小
    drop_path_rate: float  # DropPath层的丢弃率
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    def setup(self):
        # 初始化 BEiT 注意力机制
        self.attention = FlaxBeitAttention(self.config, self.window_size, dtype=self.dtype)
        # 初始化 BEiT 中间层
        self.intermediate = FlaxBeitIntermediate(self.config, dtype=self.dtype)
        # 初始化 BEiT 输出层
        self.output = FlaxBeitOutput(self.config, dtype=self.dtype)
        # 初始化 BEiT 前层标准化
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 BEiT drop path
        self.drop_path = FlaxBeitDropPath(rate=self.drop_path_rate)
        # 初始化 BEiT 后层标准化
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        # 初始化 lambda_1 和 lambda_2 参数
        self.init_values = self.config.layer_scale_init_value
        if self.init_values > 0:
            self.lambda_1 = self.param("lambda_1", ones_with_scale, (self.config.hidden_size), self.init_values)
            self.lambda_2 = self.param("lambda_2", ones_with_scale, (self.config.hidden_size), self.init_values)
        else:
            self.lambda_1 = None
            self.lambda_2 = None

    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
    ):
        # 获取自注意力输出
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 BEiT 中，自注意力前应用层标准化
            relative_position_bias,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # 如果 lambda_1 参数存在，则应用
        if self.lambda_1 is not None:
            attention_output = self.lambda_1.astype(attention_output.dtype) * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output, deterministic=deterministic) + hidden_states

        # 在 BEiT 中，自注意力后同样应用层标准化
        layer_output = self.layernorm_after(hidden_states)

        # 经过中间层处理
        layer_output = self.intermediate(layer_output)
        # 经过输出层处理
        layer_output = self.output(layer_output, deterministic=deterministic)

        # 如果 lambda_2 参数存在，则应用
        if self.lambda_2 is not None:
            layer_output = self.lambda_2.astype(layer_output.dtype) * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output, deterministic=deterministic) + hidden_states

        outputs = (layer_output,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs
class FlaxBeitLayerCollection(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    drop_path_rates: List[float]
    relative_position_bias: Callable[[], jnp.ndarray]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化层列表
        self.layers = [
            FlaxBeitLayer(
                self.config,
                window_size=self.window_size if self.config.use_relative_position_bias else None,
                drop_path_rate=self.drop_path_rates[i],
                name=str(i),
                dtype=self.dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化注意力和隐藏状态
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每一层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 获取相对位置偏置
            relative_position_bias = self.relative_position_bias() if self.relative_position_bias is not None else None
            # 调用层对象进行前向传播
            layer_outputs = layer(
                hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxBeitEncoder(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 如果使用共享的相对位置偏置，则初始化相对位置偏置对象
        if self.config.use_shared_relative_position_bias:
            self.relative_position_bias = FlaxBeitRelativePositionBias(
                config=self.config, window_size=self.window_size, dtype=self.dtype
            )

        # 根据随机深度衰减规则生成丢弃路径率列表
        drop_path_rates = list(np.linspace(0, self.config.drop_path_rate, self.config.num_hidden_layers))
        # 初始化层集合对象
        self.layer = FlaxBeitLayerCollection(
            self.config,
            window_size=self.window_size,
            drop_path_rates=drop_path_rates,
            relative_position_bias=self.relative_position_bias
            if self.config.use_shared_relative_position_bias
            else None,
            dtype=self.dtype,
        )
    # 定义 __call__ 方法，使实例对象可以像函数一样被调用
    def __call__(
        self,
        # 隐藏状态，即模型的中间输出
        hidden_states,
        # 是否确定性运行，即是否使用确定性策略
        deterministic: bool = True,
        # 是否输出注意力权重
        output_attentions: bool = False,
        # 是否输出所有隐藏状态
        output_hidden_states: bool = False,
        # 是否以字典形式返回结果
        return_dict: bool = True,
    ):
        # 调用神经网络层的方法，传入隐藏状态和其他参数，并返回结果
        return self.layer(
            hidden_states,
            # 是否确定性运行的参数
            deterministic=deterministic,
            # 是否输出注意力权重的参数
            output_attentions=output_attentions,
            # 是否输出所有隐藏状态的参数
            output_hidden_states=output_hidden_states,
            # 是否以字典形式返回结果的参数
            return_dict=return_dict,
        )
class FlaxBeitPreTrainedModel(FlaxPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 配置类
    config_class = BeitConfig
    # 基础模型前缀
    base_model_prefix = "beit"
    # 主输入名称
    main_input_name = "pixel_values"
    # 模块类
    module_class: nn.Module = None

    def __init__(
        self,
        config: BeitConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置和参数初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果未提供输入形状，则使用默认形状
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        dropout_rng, droppath_rng = jax.random.split(dropout_rng)
        # 构建随机数生成器字典
        rngs = {"params": params_rng, "dropout": dropout_rng, "droppath": droppath_rng}

        # 使用模块的初始化方法初始化参数
        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        # 如果提供了参数，则使用提供的参数，否则返回随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 如果 output_attentions 参数不为 None，则使用参数中的值；否则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数不为 None，则使用参数中的值；否则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数不为 None，则使用参数中的值；否则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 将像素值的维度转置，从 (batch_size, channels, height, width) 变为 (batch_size, height, width, channels)
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # 如果存在 dropout_rng 参数，则进行分割，将其中一部分用于 dropout，另一部分用于 droppath
        rngs = {}
        if dropout_rng is not None:
            dropout_rng, droppath_rng = jax.random.split(dropout_rng)
            rngs["dropout"] = dropout_rng
            rngs["droppath"] = droppath_rng

        # 调用模型的 apply 方法，传递参数和像素值，并根据不同的配置决定是否返回注意力权重、隐藏状态等信息
        return self.module.apply(
            {"params": params or self.params},  # 模型参数
            jnp.array(pixel_values, dtype=jnp.float32),  # 像素值，转换为浮点数类型
            bool_masked_pos,  # 是否进行遮盖
            not train,  # 是否为训练模式
            output_attentions,  # 是否输出注意力权重
            output_hidden_states,  # 是否输出隐藏状态
            return_dict,  # 是否返回结果字典
            rngs=rngs,  # 随机数生成器
        )

    # 从transformers库中导入AutoImageProcessor和FlaxBeitModel类
    from transformers import AutoImageProcessor, FlaxBeitModel
    # 从PIL库中导入Image类
    from PIL import Image
    # 导入requests模块
    import requests
    
    # 定义图像的URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过requests模块获取图像的二进制流，并使用PIL库打开该二进制流
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 使用预训练模型名称加载AutoImageProcessor类
    image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    # 使用预训练模型名称加载FlaxBeitModel类
    model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    
    # 将图像转换为模型输入，返回NumPy数组形式的张量
    inputs = image_processor(images=image, return_tensors="np")
    # 使用加载的模型处理输入数据，得到模型的输出
    outputs = model(**inputs)
    # 获取模型输出中的最后一个隐藏状态
    last_hidden_states = outputs.last_hidden_state
# 覆盖 FlaxBeitModel 的调用文档字符串为 FLAX_BEIT_MODEL_DOCSTRING 中的内容
overwrite_call_docstring(FlaxBeitModel, FLAX_BEIT_MODEL_DOCSTRING)
# 为 FlaxBeitModel 添加或替换返回的文档字符串，输出类型为 FlaxBeitModelOutputWithPooling，配置类为 BeitConfig
append_replace_return_docstrings(FlaxBeitModel, output_type=FlaxBeitModelOutputWithPooling, config_class=BeitConfig)


class FlaxBeitForMaskedImageModelingModule(nn.Module):
    # 配置属性为 BeitConfig
    config: BeitConfig
    # 计算的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化 FlaxBeitModule，不添加池化层
        self.beit = FlaxBeitModule(self.config, add_pooling_layer=False, dtype=self.dtype)

        # 分类器头部
        # LayerNorm 层，设置 epsilon 为配置中的 layer_norm_eps
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 全连接层，输出维度为配置中的 vocab_size，初始化方式为正态分布，均值为 0，标准差为配置中的 initializer_range
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 如果 return_dict 为 None，则根据配置中的 use_return_dict 决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 FlaxBeitModule 进行前向传播
        outputs = self.beit(
            pixel_values,
            bool_masked_pos,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]
        # 应用 LayerNorm 层
        sequence_output = self.layernorm(sequence_output)
        # 对序列输出进行预测，从第一个位置开始，因为第一个位置是特殊标记 [CLS]
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        # 如果不返回字典，则输出为元组形式
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output

        # 返回 FlaxMaskedLMOutput 类型的结果
        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为 FlaxBeitForMaskedImageModeling 类添加起始文档字符串
@add_start_docstrings(
    "Beit Model transformer with a 'language' modeling head on top (to predict visual tokens).",
    BEIT_START_DOCSTRING,
)
class FlaxBeitForMaskedImageModeling(FlaxBeitPreTrainedModel):
    # 模块类为 FlaxBeitForMaskedImageModelingModule
    module_class = FlaxBeitForMaskedImageModelingModule


# FlaxBeitForMaskedImageModeling 类的语言模型头部为 FLAX_BEIT_MLM_DOCSTRING 中的内容
FLAX_BEIT_MLM_DOCSTRING = """
    bool_masked_pos (`numpy.ndarray` of shape `(batch_size, num_patches)`):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

    Returns:

    Examples:

    ```py
    >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""
# 覆盖 FlaxBeitForMaskedImageModeling 类的文档字符串
overwrite_call_docstring(FlaxBeitForMaskedImageModeling, FLAX_BEIT_MLM_DOCSTRING)
# 附加替换 FlaxBeitForMaskedImageModeling 类的返回文档字符串
append_replace_return_docstrings(
    FlaxBeitForMaskedImageModeling, output_type=FlaxMaskedLMOutput, config_class=BeitConfig
)

# 定义 FlaxBeitForImageClassificationModule 类
class FlaxBeitForImageClassificationModule(nn.Module):
    # 类属性：BeitConfig 类型的 config 属性，jnp.float32 类型的 dtype 属性
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建 FlaxBeitModule 实例，传入 config 和 dtype 参数
        self.beit = FlaxBeitModule(config=self.config, dtype=self.dtype, add_pooling_layer=True)
        # 创建 nn.Dense 实例，用于分类，传入 num_labels、kernel_init 和 dtype 参数
        self.classifier = nn.Dense(
            self.config.num_labels,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法
    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.beit 方法，传入参数，获取输出
        outputs = self.beit(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]
        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)

        # 如果不需要返回字典，则返回 logits 和 outputs[2:] 组成的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        # 返回 FlaxSequenceClassifierOutput 实例，包含 logits、hidden_states 和 attentions
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 附加开始文档字符串到 FlaxBeitForImageClassification 类
@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
# 定义 FlaxBeitForImageClassification 类，继承自 FlaxBeitPreTrainedModel
class FlaxBeitForImageClassification(FlaxBeitPreTrainedModel):
    # 指定 module_class 为 FlaxBeitForImageClassificationModule
    module_class = FlaxBeitForImageClassificationModule

# FlaxBeitForImageClassification 类的文档字符串
FLAX_BEIT_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```py
    >>> from transformers import AutoImageProcessor, FlaxBeitForImageClassification
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    >>> model = FlaxBeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = logits.argmax(-1).item()
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
    ```
"""

# 覆盖 FlaxBeitForImageClassification 类的文档字符串
overwrite_call_docstring(FlaxBeitForImageClassification, FLAX_BEIT_CLASSIF_DOCSTRING)
# 附加替换 FlaxBeitForImageClassification 类的返回文档字符串
append_replace_return_docstrings(
    # 导入 FlaxBeitForImageClassification 类，指定输出类型为 FlaxSequenceClassifierOutput，配置类为 BeitConfig
    FlaxBeitForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=BeitConfig
# 该行代码为一个空行，没有实际作用，可以忽略
```