# `.\models\vit\modeling_flax_vit.py`

```
#python
# 导入必要的模块和类
from typing import Optional, Tuple

import flax.linen as nn  # 导入 Flax 的 linen 模块
import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的 FrozenDict 相关函数
from flax.linen.attention import dot_product_attention_weights  # 导入 dot_product_attention_weights 函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 flatten_dict 和 unflatten_dict 函数

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput  # 导入输出相关类
from ...modeling_flax_utils import (  # 导入 FlaxPreTrainedModel 和其他实用函数
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward  # 导入添加文档字符串的函数
from .configuration_vit import ViTConfig  # 导入 ViTConfig 配置类

VIT_START_DOCSTRING = r"""

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
    # Parameters: 定义函数的参数列表及其描述
    # config ([`ViTConfig`]): 使用ViTConfig类配置模型的参数
    #     通过配置文件初始化不会加载模型的权重，仅加载配置。请查看[`~FlaxPreTrainedModel.from_pretrained`]方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    #     计算时所使用的数据类型。可以是`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和`jax.numpy.bfloat16`（在TPU上）之一。
    #     
    #     可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的`dtype`执行。
    #     
    #     **请注意，此仅指定计算的数据类型，不影响模型参数的数据类型。**
    #     
    #     如果您希望更改模型参数的数据类型，请参阅[`~FlaxPreTrainedModel.to_fp16`]和[`~FlaxPreTrainedModel.to_bf16`]。
"""
Define a docstring for the module `FlaxViTPatchEmbeddings`.

This docstring provides detailed information about the inputs expected by the module, including:
- `pixel_values`: A numpy array containing pixel values of shape `(batch_size, num_channels, height, width)`.
  This array can be obtained using an `AutoImageProcessor`.
- `output_attentions`: An optional boolean indicating whether to return attention tensors from all layers.
- `output_hidden_states`: An optional boolean indicating whether to return hidden states from all layers.
- `return_dict`: An optional boolean indicating whether to return a `ModelOutput` instead of a tuple.
"""
class FlaxViTPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # Calculate number of patches in the image
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        # Initialize a convolutional layer for projecting patches
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(self, pixel_values):
        # Check if the number of channels in pixel values matches the configuration
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # Project pixel values into patch embeddings
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        # Reshape embeddings to match the expected output format
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 在模型设置阶段初始化特殊的“CLS”标记，它是一个参数，根据给定的初始化器创建
    # 使用方差缩放初始化器（以配置的初始化范围的平方为参数），初始化“CLS”标记
    self.cls_token = self.param(
        "cls_token",
        jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
        (1, 1, self.config.hidden_size),
    )

    # 创建图像块的嵌入表示对象，使用ViT中的补丁嵌入（Patch Embeddings）
    self.patch_embeddings = FlaxViTPatchEmbeddings(self.config, dtype=self.dtype)

    # 计算图像块的数量，并初始化位置嵌入，将其视为模型参数
    num_patches = self.patch_embeddings.num_patches
    self.position_embeddings = self.param(
        "position_embeddings",
        jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
        (1, num_patches + 1, self.config.hidden_size),
    )

    # 初始化一个dropout层，根据配置中的隐藏层dropout概率
    self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

def __call__(self, pixel_values, deterministic=True):
    # 获取输入张量的批次大小
    batch_size = pixel_values.shape[0]

    # 对输入的像素值计算补丁嵌入（Patch Embeddings）
    embeddings = self.patch_embeddings(pixel_values)

    # 创建形状与批次大小匹配的“CLS”标记，并将其广播到嵌入的第二个维度
    cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))

    # 将“CLS”标记与图像块嵌入连接起来，形成完整的嵌入表示
    embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

    # 将位置嵌入添加到嵌入向量中，以增强位置信息
    embeddings = embeddings + self.position_embeddings

    # 应用dropout，用于模型训练时的随机失活，以防止过拟合
    embeddings = self.dropout(embeddings, deterministic=deterministic)

    # 返回最终的嵌入表示作为模型的输出
    return embeddings
# 定义一个名为 FlaxViTSelfAttention 的自定义 PyTorch 模块
class FlaxViTSelfAttention(nn.Module):
    # ViT 模型的配置信息
    config: ViTConfig
    # 计算时所使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块的初始化方法
    def setup(self):
        # 检查 hidden_size 是否能被 num_attention_heads 整除
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        # 初始化查询（query）的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        # 初始化键（key）的全连接层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        # 初始化值（value）的全连接层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    # 模块的调用方法，实现自注意力机制
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 使用查询向量进行全连接操作并重塑形状
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用值向量进行全连接操作并重塑形状
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用键向量进行全连接操作并重塑形状
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 初始化一个用于 dropout 的随机数生成器
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算自注意力权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 使用注意力权重和值向量计算最终的自注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # 重塑输出的形状
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 如果需要输出注意力权重，将其包含在输出中
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# 定义一个名为 FlaxViTSelfOutput 的自定义 PyTorch 模块
class FlaxViTSelfOutput(nn.Module):
    # ViT 模型的配置信息
    config: ViTConfig
    # 计算时所使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义类的初始化方法，用于设置模型参数
    def setup(self):
        # 初始化一个全连接层，设置输出大小为self.config.hidden_size
        # 使用方差缩放初始化方法，基于截断的正态分布，参数为self.config.initializer_range的平方
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        # 初始化一个Dropout层，设置丢弃率为self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 定义类的调用方法，用于模型推理过程
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 将输入的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用Dropout层，用于随机丢弃部分神经元的输出
        # 根据deterministic参数决定是否以确定性方式进行操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回经过全连接层和Dropout层处理后的隐藏状态作为最终的模型输出
        return hidden_states
class FlaxViTAttention(nn.Module):
    config: ViTConfig  # 类属性，指定 ViT 的配置
    dtype: jnp.dtype = jnp.float32  # 类属性，默认使用 jnp.float32 数据类型

    def setup(self):
        self.attention = FlaxViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)
        # 初始化 self.attention 和 self.output，使用指定的配置和数据类型

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        # 调用 self.attention 对象处理 hidden_states，获取注意力输出
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        # 使用 self.output 处理注意力输出和 hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)
            # 如果需要输出注意力信息，则将注意力权重信息添加到输出中

        return outputs
        # 返回处理后的 hidden_states 和可能的注意力信息


class FlaxViTIntermediate(nn.Module):
    config: ViTConfig  # 类属性，指定 ViT 的配置
    dtype: jnp.dtype = jnp.float32  # 类属性，默认使用 jnp.float32 数据类型

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        # 初始化 self.dense，使用指定的中间层大小和初始化方法

        self.activation = ACT2FN[self.config.hidden_act]
        # 根据配置选择激活函数

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # 使用 self.dense 处理 hidden_states
        hidden_states = self.activation(hidden_states)
        # 使用选择的激活函数处理 hidden_states
        return hidden_states
        # 返回处理后的 hidden_states


class FlaxViTOutput(nn.Module):
    config: ViTConfig  # 类属性，指定 ViT 的配置
    dtype: jnp.dtype = jnp.float32  # 类属性，默认使用 jnp.float32 数据类型

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        # 初始化 self.dense，使用指定的输出层大小和初始化方法

        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化 self.dropout，使用指定的 dropout 比率

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        # 使用 self.dense 处理 hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 使用 self.dropout 处理 hidden_states
        hidden_states = hidden_states + attention_output
        # 将处理后的 hidden_states 与 attention_output 相加作为最终输出
        return hidden_states
        # 返回处理后的 hidden_states


class FlaxViTLayer(nn.Module):
    config: ViTConfig  # 类属性，指定 ViT 的配置
    dtype: jnp.dtype = jnp.float32  # 类属性，默认使用 jnp.float32 数据类型

    def setup(self):
        self.attention = FlaxViTAttention(self.config, dtype=self.dtype)
        # 初始化 self.attention，使用指定的配置和数据类型
        self.intermediate = FlaxViTIntermediate(self.config, dtype=self.dtype)
        # 初始化 self.intermediate，使用指定的配置和数据类型
        self.output = FlaxViTOutput(self.config, dtype=self.dtype)
        # 初始化 self.output，使用指定的配置和数据类型
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 self.layernorm_before，使用指定的层归一化参数和数据类型
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 self.layernorm_after，使用指定的层归一化参数和数据类型
    # 定义一个调用方法，用于处理隐藏状态，接收是否确定性处理的参数和是否输出注意力的参数
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        # 使用自注意力机制处理前的层归一化，这是 ViT 中的操作顺序
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 ViT 中，自注意力前会进行层归一化
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # 第一个残差连接
        attention_output = attention_output + hidden_states

        # 在 ViT 中，自注意力后同样会进行层归一化
        layer_output = self.layernorm_after(attention_output)

        # 经过中间层的处理
        hidden_states = self.intermediate(layer_output)

        # 输出层的最终处理，同时传入注意力输出和隐藏状态的处理结果
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        # 如果需要输出注意力的话，将注意力输出加入到返回的元组中
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        # 返回处理后的输出结果元组
        return outputs
# 定义一个 FlaxViTLayerCollection 类，继承自 nn.Module
class FlaxViTLayerCollection(nn.Module):
    # 定义类变量 config，类型为 ViTConfig
    config: ViTConfig
    # 定义 dtype 变量，默认为 jnp.float32，用于计算的数据类型

    # 初始化函数，设置网络层集合
    def setup(self):
        # 创建 self.layers 列表，包含 self.config.num_hidden_layers 个 FlaxViTLayer 实例
        self.layers = [
            FlaxViTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 实现 __call__ 方法，用于模型的前向传播
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态张量
        deterministic: bool = True,  # 是否使用确定性计算，默认为 True
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典格式的输出，默认为 True
    ):
        # 如果 output_attentions 为 True，则初始化 all_attentions 为空元组，否则为 None
        all_attentions = () if output_attentions else None
        # 如果 output_hidden_states 为 True，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历 self.layers 中的每一层
        for i, layer in enumerate(self.layers):
            # 如果输出所有隐藏状态
            if output_hidden_states:
                # 将当前隐藏状态加入 all_hidden_states 中
                all_hidden_states += (hidden_states,)

            # 调用当前层的 __call__ 方法，进行前向传播
            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            # 更新 hidden_states 为当前层的输出的第一个元素（即隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重加入 all_attentions 中
                all_attentions += (layer_outputs[1],)

        # 如果输出所有隐藏状态
        if output_hidden_states:
            # 将最终的隐藏状态加入 all_hidden_states 中
            all_hidden_states += (hidden_states,)

        # 将最终的隐藏状态作为元组 outputs 的第一个元素
        outputs = (hidden_states,)

        # 如果不返回字典格式的输出
        if not return_dict:
            # 返回 outputs 中非 None 的元素作为元组
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 类的实例，包含最终的隐藏状态、所有隐藏状态和注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# 定义一个 FlaxViTEncoder 类，继承自 nn.Module
class FlaxViTEncoder(nn.Module):
    # 定义类变量 config，类型为 ViTConfig
    config: ViTConfig
    # 定义 dtype 变量，默认为 jnp.float32，用于计算的数据类型

    # 初始化函数，创建 FlaxViTLayerCollection 类的实例作为 self.layer
    def setup(self):
        self.layer = FlaxViTLayerCollection(self.config, dtype=self.dtype)

    # 实现 __call__ 方法，用于模型的前向传播
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态张量
        deterministic: bool = True,  # 是否使用确定性计算，默认为 True
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典格式的输出，默认为 True
    ):
        # 调用 self.layer 的 __call__ 方法进行前向传播
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 定义一个 FlaxViTPooler 类，继承自 nn.Module
class FlaxViTPooler(nn.Module):
    # 定义类变量 config，类型为 ViTConfig
    config: ViTConfig
    # 定义 dtype 变量，默认为 jnp.float32，用于计算的数据类型

    # 初始化函数，设置池化层为全连接层
    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    # 实现 __call__ 方法，用于模型的前向传播
    def __call__(self, hidden_states):
        # 取出每个样本的第一个位置的隐藏状态作为池化结果
        cls_hidden_state = hidden_states[:, 0]
        # 将 cls_hidden_state 输入到全连接层中
        cls_hidden_state = self.dense(cls_hidden_state)
        # 对全连接层的输出进行 tanh 激活函数处理
        return nn.tanh(cls_hidden_state)


# 定义一个 FlaxViTPreTrainedModel 类，继承自 FlaxPreTrainedModel
class FlaxViTPreTrainedModel(FlaxPreTrainedModel):
    """
    一个处理权重初始化和简单接口以下载和加载预训练模型的抽象类。
    """

    # 类变量，指定配置类为 ViTConfig
    config_class = ViTConfig
    # 模型的基础名称前缀为 "vit"
    base_model_prefix = "vit"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 模型类变量，默认为 None
    module_class: nn.Module = None

    # 初始化函数，接受配置对象、输入形状、种子值、数据类型和其他关键字参数
    def __init__(
        self,
        config: ViTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和数据类型实例化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        
        # 如果未提供输入形状，则设置为默认的图像大小和通道数
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        
        # 调用父类的初始化方法，传递配置、模块、输入形状、种子、数据类型和是否初始化参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化模型权重的函数，接受随机数生成器、输入形状和参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全零张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 切分随机数生成器为参数随机数和 dropout 随机数
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块对象的初始化方法初始化随机参数
        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        # 如果传入了已有参数，则将缺失的参数从随机参数中添加进去
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 调用函数，接受像素值、参数字典、dropout 随机数生成器、训练标志、是否输出注意力、隐藏状态和是否返回字典
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 如果未指定输出注意力，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 转置像素值的维度，将通道维度移到最后一个位置
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        
        # 如果需要处理 dropout 的随机数生成器，则添加到随机数字典中
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 应用模块对象的前向传播函数，传入参数字典或者自身的参数、像素值、训练标志、输出注意力、隐藏状态、返回字典和随机数字典
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
class FlaxViTModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        # 初始化 ViT 模型的嵌入层
        self.embeddings = FlaxViTEmbeddings(self.config, dtype=self.dtype)
        # 初始化 ViT 模型的编码器层
        self.encoder = FlaxViTEncoder(self.config, dtype=self.dtype)
        # 初始化 LayerNorm 层，用于规范化隐藏状态
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 如果设置了添加池化层选项，初始化 ViT 模型的池化层
        self.pooler = FlaxViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 将像素值转换为隐藏状态向量
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        # 使用编码器处理隐藏状态向量，获取输出
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的输出的隐藏状态向量
        hidden_states = outputs[0]
        # 对隐藏状态向量进行 LayerNorm 规范化
        hidden_states = self.layernorm(hidden_states)
        # 如果设置了池化层且存在，对隐藏状态向量进行池化操作
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不返回字典，根据池化层的存在性选择性返回结果
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回包含池化输出的 FlaxBaseModelOutputWithPooling 对象
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class FlaxViTModel(FlaxViTPreTrainedModel):
    module_class = FlaxViTModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxViTModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    >>> model = FlaxViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxViTModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxViTModel, output_type=FlaxBaseModelOutputWithPooling, config_class=ViTConfig)


class FlaxViTForImageClassificationModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32
    # 在对象初始化时设置模型结构，使用指定的配置和数据类型，不添加池化层
    def setup(self):
        self.vit = FlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        
        # 初始化分类器，设置输出类别数和数据类型，并使用截断正态分布初始化权重
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    # 实现对象的调用功能，接受像素值、确定性标志、是否输出注意力和隐藏状态、返回字典等参数
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 如果没有显式指定返回字典的用法，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练的 ViT 模型进行前向传播
        outputs = self.vit(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出的隐藏状态，并通过分类器获取对应的 logits
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states[:, 0, :])

        # 如果不需要返回字典形式的输出，则组装为元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包括 logits 和额外的隐藏状态
            return output

        # 返回格式化后的分类器输出对象，包括 logits、隐藏状态和注意力权重
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 在FlaxViTForImageClassification类之前添加文档字符串，描述其作为ViT模型转换器的用途，顶部有一个基于图像分类的线性层的简要说明，例如用于ImageNet。
@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    VIT_START_DOCSTRING,  # 添加ViT模型的起始文档字符串
)

# 指定FlaxViTForImageClassification的模块类为FlaxViTForImageClassificationModule
class FlaxViTForImageClassification(FlaxViTPreTrainedModel):
    module_class = FlaxViTForImageClassificationModule


# FLAX_VISION_CLASSIF_DOCSTRING 是一个多行字符串，用于描述FlaxViTForImageClassification类的返回值和示例
FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxViTForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    >>> model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

# 使用overwrite_call_docstring函数，将FLAX_VISION_CLASSIF_DOCSTRING的内容覆盖到FlaxViTForImageClassification类的文档字符串中
overwrite_call_docstring(FlaxViTForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)

# 使用append_replace_return_docstrings函数，将输出类型设置为FlaxSequenceClassifierOutput，并指定配置类为ViTConfig
append_replace_return_docstrings(
    FlaxViTForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=ViTConfig
)
```