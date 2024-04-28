# `.\transformers\models\vit\modeling_flax_vit.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明代码版权归 The Google Flax Team Authors 和 The HuggingFace Inc. team 所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit import ViTConfig

# ViT 模型的起始文档字符串
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
``` 
    # 参数:
    # config ([`ViTConfig`]): 包含模型所有参数的模型配置类。
    # 初始化时使用配置文件不会加载与模型相关的权重，只会加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    # 计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` (在GPU上) 和 `jax.numpy.bfloat16` (在TPU上) 中的一种。
    # 这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，所有计算将使用给定的 `dtype` 进行。
    # **请注意，这仅指定计算的数据类型，不会影响模型参数的数据类型。**
    # 如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxViTPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
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
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化模型参数
    def setup(self):
        # 初始化 cls_token 参数，用于表示类别信息
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        # 初始化 patch_embeddings，用于将像素值转换为嵌入向量
        self.patch_embeddings = FlaxViTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        # 初始化 position_embeddings，用于表示位置信息
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        # 初始化 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 模型的前向传播
    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]

        # 获取像素值的嵌入向量
        embeddings = self.patch_embeddings(pixel_values)

        # 创建 cls_tokens，用于表示类别信息，并与嵌入向量拼接
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        # 添加位置嵌入向量
        embeddings = embeddings + self.position_embeddings
        # 对嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        # 返回处理后的嵌入向量
        return embeddings
# 定义一个自注意力模块，继承自 nn.Module
class FlaxViTSelfAttention(nn.Module):
    # ViT 模型的配置
    config: ViTConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  

    # 模块的初始化方法
    def setup(self):
        # 检查隐藏层大小是否是注意力头数的倍数
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        # 创建查询、键、值的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    # 自注意力模块的调用方法
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 计算查询、键、值的结果并重塑形状
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 初始化 dropout_rng
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

        # 计算自注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 如果需要输出注意力权重，则返回注意力输出和注意力权重，否则只返回注意力输出
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# 定义 ViT 自注意力模块的输出层
class FlaxViTSelfOutput(nn.Module):
    # ViT 模型的配置
    config: ViTConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化模型参数
    def setup(self):
        # 初始化稠密层，指定隐藏层大小和初始化方式
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                # 使用方差缩放初始化方法，参数为隐藏层大小的平方和 fan_in 方式
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            # 指定数据类型
            dtype=self.dtype,
        )
        # 初始化丢弃层，指定丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 对象被调用时执行的方法
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 稠密层前向传播，计算隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用丢弃层进行丢弃操作，传入隐藏状态和是否确定性参数
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回处理后的隐藏状态
        return hidden_states
class FlaxViTAttention(nn.Module):
    config: ViTConfig  # 定义 ViT 模型的配置
    dtype: jnp.dtype = jnp.float32  # 定义数据类型为 jnp.float32

    def setup(self):
        self.attention = FlaxViTSelfAttention(self.config, dtype=self.dtype)  # 初始化自注意力层
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)  # 初始化自注意力输出层

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)  # 调用自注意力层
        attn_output = attn_outputs[0]  # 获取自注意力输出
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)  # 调用自注意力输出层

        outputs = (hidden_states,)  # 将隐藏状态存入元组

        if output_attentions:
            outputs += (attn_outputs[1],)  # 如果需要输出注意力权重，则将注意力权重存入元组

        return outputs  # 返回结果元组


class FlaxViTIntermediate(nn.Module):
    config: ViTConfig  # 定义 ViT 模型的配置
    dtype: jnp.dtype = jnp.float32  # 定义数据类型为 jnp.float32

    def setup(self):
        self.dense = nn.Dense(  # 初始化全连接层
            self.config.intermediate_size,  # 中间层大小
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),  # 初始化权重
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]  # 激活函数

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 全连接层计算
        hidden_states = self.activation(hidden_states)  # 激活函数
        return hidden_states  # 返回结果


class FlaxViTOutput(nn.Module):
    config: ViTConfig  # 定义 ViT 模型的配置
    dtype: jnp.dtype = jnp.float32  # 定义数据类型为 jnp.float32

    def setup(self):
        self.dense = nn.Dense(  # 初始化全连接层
            self.config.hidden_size,  # 隐藏层大小
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),  # 初始化权重
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)  # 初始化 Dropout 层

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)  # 全连接层计算
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # Dropout 操作
        hidden_states = hidden_states + attention_output  # 加上注意力输出
        return hidden_states  # 返回结果


class FlaxViTLayer(nn.Module):
    config: ViTConfig  # 定义 ViT 模型的配置
    dtype: jnp.dtype = jnp.float32  # 定义数据类型为 jnp.float32

    def setup(self):
        self.attention = FlaxViTAttention(self.config, dtype=self.dtype)  # 初始化注意力层
        self.intermediate = FlaxViTIntermediate(self.config, dtype=self.dtype)  # 初始化中间层
        self.output = FlaxViTOutput(self.config, dtype=self.dtype)  # 初始化输出层
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 初始化 LayerNorm 层
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 初始化 LayerNorm 层
    # 定义一个调用函数，接受隐藏状态、确定性标志和输出注意力的布尔值作为参数
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        # 使用 self-attention 处理隐藏状态，返回注意力输出
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 ViT 中，layernorm 在 self-attention 之前应用
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # 第一个残差连接
        attention_output = attention_output + hidden_states

        # 在 ViT 中，layernorm 也在 self-attention 之后应用
        layer_output = self.layernorm_after(attention_output)

        # 使用 intermediate 层处理输出
        hidden_states = self.intermediate(layer_output)
        # 使用 output 层处理 hidden_states，并返回结果
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        # 如果需要输出注意力，将注意力输出添加到结果中
        if output_attentions:
            outputs += (attention_outputs[1],)
        # 返回结果
        return outputs
class FlaxViTLayerCollection(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化层集合，根据配置创建多个 ViT 层
        self.layers = [
            FlaxViTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化存储注意力和隐藏状态的变量
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个 ViT 层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用 ViT 层，获取输出
            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

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


class FlaxViTEncoder(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化 ViT 编码器，包含 ViT 层集合
        self.layer = FlaxViTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 ViT 层集合
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxViTPooler(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化池化层，包含一个全连接层
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # 获取第一个位置的隐藏状态，通过全连接层得到池化结果
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxViTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    # 定义一个类变量 module_class，类型为 nn.Module，默认为 None
    module_class: nn.Module = None

    # 初始化函数，接受配置参数 config、输入形状 input_shape、种子数 seed、数据类型 dtype、是否初始化 _do_init 和其他关键字参数
    def __init__(
        self,
        config: ViTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置参数和其他关键字参数创建模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果输入形状未指定，则设置默认输入形状
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类的初始化函数，传入配置参数、模块对象、输入形状、种子数、数据类型和是否初始化参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重函数，接受随机数种子 rng、输入形状 input_shape、参数 params，默认为 None，返回冻结的参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 pixel_values，全零张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 分割随机数种子，得到参数随机数和 dropout 随机数
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化随机参数
        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        # 如果参数不为 None，则将随机参数和参数展平后进行合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 模型调用函数，接受像素值 pixel_values、参数 params，默认为 None，dropout 随机数 dropout_rng，默认为 None，训练标志 train，默认为 False，输出注意力 output_attentions，默认为 None，输出隐藏状态 output_hidden_states，默认为 None，返回字典标志 return_dict，默认为 None
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
        # 如果输出注意力未指定，则使用配置参数中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置参数中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典标志未指定，则使用配置参数中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 调整像素值的维度顺序
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # 处理任何需要的 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的应用方法，传入参数、像素值、是否训练、输出注意力、输出隐藏状态、返回字典标志和随机数字典
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
# 创建名为FlaxViTModule的类，继承自nn.Module
class FlaxViTModule(nn.Module):
    # 声明config为ViTConfig类型的属性
    config: ViTConfig
    # 设置dtype为jnp.float32，作为计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 设置add_pooling_layer为True
    add_pooling_layer: bool = True

    # 初始化方法
    def setup(self):
        # 初始化embeddings属性为FlaxViTEmbeddings对象，传入config和dtype
        self.embeddings = FlaxViTEmbeddings(self.config, dtype=self.dtype)
        # 初始化encoder属性为FlaxViTEncoder对象，传入config和dtype
        self.encoder = FlaxViTEncoder(self.config, dtype=self.dtype)
        # 初始化layernorm属性为nn.LayerNorm对象，传入config.layer_norm_eps和dtype
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 如果add_pooling_layer为True，则初始化pooler属性为FlaxViTPooler对象，传入config和dtype
        self.pooler = FlaxViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    # 调用方法
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 使用embeddings处理pixel_values，传入deterministic
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        # 使用encoder处理hidden_states，传入deterministic、output_attentions、output_hidden_states、return_dict
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取outputs的第一个元素，并赋值给hidden_states
        hidden_states = outputs[0]
        # 使用layernorm处理hidden_states
        hidden_states = self.layernorm(hidden_states)
        # 如果add_pooling_layer为True，则使用pooler处理hidden_states，并赋值给pooled；否则pooled为None
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果return_dict为False
        if not return_dict:
            # 如果pooled为None，则返回元组(hidden_states, outputs[1:])；否则返回元组(hidden_states, pooled, outputs[1:])
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回FlaxBaseModelOutputWithPooling对象
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 将FlaxViTModel类的文档字符串替换为指定内容
@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class FlaxViTModel(FlaxViTPreTrainedModel):
    module_class = FlaxViTModule


# 定义FLAX_VISION_MODEL_DOCSTRING字符串
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

# 覆盖FlaxViTModel类的调用方法文档字符串
overwrite_call_docstring(FlaxViTModel, FLAX_VISION_MODEL_DOCSTRING)
# 在FlaxViTModel类的返回文档字符串中追加或替换指定内容
append_replace_return_docstrings(FlaxViTModel, output_type=FlaxBaseModelOutputWithPooling, config_class=ViTConfig)


# 创建名为FlaxViTForImageClassificationModule的类，继承自nn.Module
class FlaxViTForImageClassificationModule(nn.Module):
    # 声明config为ViTConfig类型的属性
    config: ViTConfig
    # 设置dtype为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 初始化模型参数和配置
    def setup(self):
        # 使用配置和数据类型构建FlaxViTModule，禁用添加池化层
        self.vit = FlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        # 构建一个全连接层分类器
        self.classifier = nn.Dense(
            self.config.num_labels,  # 输出类别数
            dtype=self.dtype,  # 数据类型
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"  # 初始化权重的方法和参数
            ),
        )

    # 模型的调用方法
    def __call__(
        self,
        pixel_values=None,  # 输入像素值
        deterministic: bool = True,  # 是否确定性
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 返回结果的类型
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 设置return_dict的默认值

        # 调用ViT模块获取输出结果
        outputs = self.vit(
            pixel_values,  # 输入像素值
            deterministic=deterministic,  # 是否确定性
            output_attentions=output_attentions,  # 输出注意力权重
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 返回结果的类型
        )

        hidden_states = outputs[0]  # 从输出中获取隐藏状态
        logits = self.classifier(hidden_states[:, 0, :])  # 使用分类器对隐藏状态进行分类预测

        if not return_dict:
            # 如果不需要返回字典类型的结果，则返回一个元组
            output = (logits,) + outputs[2:]
            return output

        # 如果需要返回字典类型的结果，则构建一个FlaxSequenceClassifierOutput对象并返回
        return FlaxSequenceClassifierOutput(
            logits=logits,  # 分类预测
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )
# 为 FlaxViTForImageClassification 类添加文档字符串，描述其在图像分类任务中的用途
@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    VIT_START_DOCSTRING,
)
class FlaxViTForImageClassification(FlaxViTPreTrainedModel):
    # 指定模块类为 FlaxViTForImageClassificationModule
    module_class = FlaxViTForImageClassificationModule


# 添加 FlaxViTForImageClassification 类的文档字符串，说明其返回的输出和示例用法
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

# 为 FlaxViTForImageClassification 类的调用方法添加文档字符串，指定输出类型和配置类
overwrite_call_docstring(FlaxViTForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
append_replace_return_docstrings(
    FlaxViTForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=ViTConfig
)
```