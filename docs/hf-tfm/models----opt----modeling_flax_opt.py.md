# `.\transformers\models\opt\modeling_flax_opt.py`

```
# 导入必要的库和模块
from functools import partial  # 导入 partial 函数，用于创建偏函数
from typing import Optional, Tuple  # 导入类型提示模块

# 导入 Flax 模块和函数
import flax.linen as nn  # 导入 flax.linen 模块，并重命名为 nn
import jax  # 导入 jax 库
import jax.numpy as jnp  # 导入 jax.numpy 模块，并重命名为 jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 FrozenDict 相关函数
from flax.linen import combine_masks, make_causal_mask  # 导入 combine_masks 和 make_causal_mask 函数
from flax.linen.attention import dot_product_attention_weights  # 导入 dot_product_attention_weights 函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 flatten_dict 和 unflatten_dict 函数
from jax import lax  # 导入 lax 模块
from jax.random import PRNGKey  # 导入 PRNGKey 类

# 导入模型输出和工具函数
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput  # 导入模型输出类
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring  # 导入模型工具函数和常量
from ...utils import add_start_docstrings, logging  # 导入添加文档字符串函数和日志记录模块

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/opt-350m"  # 检查点路径
_CONFIG_FOR_DOC = "OPTConfig"  # 配置信息

# OPT 模型的文档起始部分
OPT_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
``` 
    # 参数说明：
        # config: 模型配置类，包含模型的所有参数。
            # 初始化时使用配置文件不会加载与模型关联的权重，只加载配置。
            # 查看`~FlaxPreTrainedModel.from_pretrained`方法以加载模型权重。
        # dtype (`jax.numpy.dtype`, *可选*, 默认为`jax.numpy.float32`):
            # 计算的数据类型。可以是`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和`jax.numpy.bfloat16`（在TPU上）之一。
            
            # 这可以用于在GPU或TPU上启用混合精度训练或半精度推理。如果指定，所有计算将使用给定的`dtype`进行。
            
            # **请注意，这只指定了计算的dtype，并不影响模型参数的dtype。**
            
            # 如果希望更改模型参数的dtype，请参阅[`~FlaxPreTrainedModel.to_fp16`]和[`~FlaxPreTrainedModel.to_bf16`]。
# 定义 OPT 模型输入的文档字符串
OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列的词汇表索引。默认情况下会忽略填充。

            可以使用 [`AutoTokenizer`] 获得索引。查看 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 了解详情。

            [什么是 input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            注意力掩码，避免在填充令牌上进行注意力计算。掩码值在 `[0, 1]` 中选择:

            - 1 表示 **未被掩盖** 的令牌，
            - 0 表示 **被掩盖** 的令牌。

            [什么是注意力掩码?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列令牌在位置嵌入中的位置索引。取值范围为 `[0, config.max_position_embeddings - 1]`。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。更多详情请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。更多详情请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制，将 Bart 更改为 OPT
class FlaxOPTAttention(nn.Module):
    # 配置对象
    config: OPTConfig
    # 嵌入维度
    embed_dim: int
    # 注意力头数
    num_heads: int
    # 丢弃率
    dropout: float = 0.0
    # 因果注意力
    causal: bool = False
    # 是否使用偏置
    bias: bool = True
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果嵌入维度不能被注意力头数整除则报错
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 使用 nn.Dense 创建三个线性层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 创建丢弃层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果是因果注意力，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )
    # 将隐藏状态分割成多个头部，用于多头注意力机制
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将头部合并成隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 将投影的键、值状态从单个输入标记拼接到先前步骤的缓存状态
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        此函数将投影的键、值状态从单个输入标记拼接到先前步骤的缓存状态。
        此函数略微改编自官方 Flax 存储库：https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的 1 维空间切片更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 缓存的解码器自注意力的因果蒙版：我们的单个查询位置应该只与已生成和缓存的键位置进行注意力，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    # 调用函数
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
# 定义 FlaxOPTDecoderLayer 类，继承自 nn.Module
class FlaxOPTDecoderLayer(nn.Module):
    # 定义类属性 config，类型为 OPTConfig
    config: OPTConfig
    # 定义类属性 dtype，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义 setup 方法，用于初始化层
    def setup(self) -> None:
        # 设置嵌入维度为隐藏大小
        self.embed_dim = self.config.hidden_size
        # 初始化 self_attn 层，使用 FlaxOPTAttention 类
        self.self_attn = FlaxOPTAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 是否在层归一化之前应用 self-attention
        self.do_layer_norm_before = self.config.do_layer_norm_before
        # 定义 dropout 层，用于在训练时进行随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数选择，根据配置选择相应的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]

        # 初始化 self_attn 层之后的层归一化层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 第一个全连接层，用于处理 self_attn 输出
        self.fc1 = nn.Dense(
            self.config.ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，用于将处理后的输出映射回原始维度
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的层归一化层，用于处理最终的输出
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义 __call__ 方法，用于调用层
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    # 定义了一个方法，接受隐藏状态作为输入，并返回包含隐藏状态的元组
    ) -> Tuple[jnp.ndarray]:
        # 将隐藏状态赋值给残差变量
        residual = hidden_states

        # 如果设置了在注意力之前进行层归一化
        if self.do_layer_norm_before:
            # 在注意力之前对隐藏状态进行层归一化
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
        )
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差和新的隐藏状态相加
        hidden_states = residual + hidden_states
        # 如果没有设置在注意力之前进行层归一化
        if not self.do_layer_norm_before:
            # 在注意力之后对隐藏状态进行层归一化
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 全连接层
        hidden_states_shape = hidden_states.shape
        # 将隐藏状态展平
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # 将隐藏状态赋值给残差变量
        residual = hidden_states

        # 如果设置了在注意力之前进行层归一化
        if self.do_layer_norm_before:
            # 在注意力之前对隐藏状态进行层归一化
            hidden_states = self.final_layer_norm(hidden_states)

        # 应用第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)

        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将残差和新的隐藏状态相加，并将形状还原
        hidden_states = (residual + hidden_states).reshape(hidden_states_shape)

        # 如果没有设置在注意力之前进行层归一化
        if not self.do_layer_norm_before:
            # 在注意力之后对隐藏状态进行层归一化
            hidden_states = self.final_layer_norm(hidden_states)

        # 将隐藏状态添加到输出中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            # 将自注意力权重添加到输出中
            outputs += (self_attn_weights,)

        # 返回输出
        return outputs
# 定义一个名为 FlaxOPTDecoderLayerCollection 的类，继承自 nn.Module
class FlaxOPTDecoderLayerCollection(nn.Module):
    # 定义 config 属性，类型为 OPTConfig
    config: OPTConfig
    # 定义 dtype 属性，默认为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 设置方法
    def setup(self):
        # 创建一个包含多个 FlaxOPTDecoderLayer 实例的列表，根据 config 中的 num_hidden_layers 进行循环创建
        self.layers = [
            FlaxOPTDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
        # 设置 layerdrop 属性为 config 中的 layerdrop
        self.layerdrop = self.config.layerdrop

    # 调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # 如果需要输出隐藏状态，则初始化一个空的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力分布，则初始化一个空的元组
        all_self_attns = () if output_attentions else None

        # 遍历所有的 decoder layers
        for decoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前的隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前 decoder_layer 实例
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                init_cache=init_cache,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )

            # 更新当前的 hidden_states 为当前 decoder_layer 的输出中的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力分布，则将当前的注意力分布加入到 all_self_attns 中
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 返回包含 hidden_states、all_hidden_states 和 all_self_attns 的列表
        outputs = [hidden_states, all_hidden_states, all_self_attns]
        return outputs


# 定义一个名为 FlaxOPTLearnedPositionalEmbedding 的类，继承自 nn.Embed
class FlaxOPTLearnedPositionalEmbedding(nn.Embed):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    # 设置方法
    def setup(self):
        # 设置 offset 属性为 2
        self.offset = 2
        # 创建一个 param，表示位置嵌入，形状为 (self.num_embeddings + self.offset, self.features)，数据类型为 param_dtype
        self.embedding = self.param(
            "embedding", self.embedding_init, (self.num_embeddings + self.offset, self.features), self.param_dtype
        )

    # 调用方法
    def __call__(self, positions):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        # 调用父类的 __call__ 方法，输入参数为 positions + self.offset
        return super().__call__(positions + self.offset)


# 定义一个名为 FlaxOPTDecoder 的类，继承自 nn.Module
class FlaxOPTDecoder(nn.Module):
    # 定义 config 属性，类型为 OPTConfig
    config: OPTConfig
    # 定义 dtype 属性，默认为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义 offset 属性，默认为 2
    offset: int = 2
    # 设置模型的初始化参数
    def setup(self):
        # 创建一个dropout层，根据配置中的dropout rate进行dropout操作
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
    
        # 设置嵌入维度、填充token的id及最大目标位置长度
        embed_dim = self.config.hidden_size
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
    
        # 创建词嵌入层，使用正态分布初始化
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.word_embed_proj_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
    
        # 创建位置嵌入层，使用正态分布初始化
        self.embed_positions = FlaxOPTLearnedPositionalEmbedding(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
    
        # 如果词嵌入维度与隐藏层维度不同，添加输入和输出的投影层
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = nn.Dense(self.config.hidden_size, use_bias=False)
            self.project_out = nn.Dense(self.config.word_embed_proj_dim, use_bias=False)
        else:
            self.project_in = None
            self.project_out = None
    
        # 根据配置决定是否添加最终的LayerNorm层
        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        else:
            self.final_layer_norm = None
    
        # 创建解码器层集合
        self.layers = FlaxOPTDecoderLayerCollection(self.config, self.dtype)
    
    # 模型前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        ):
        # 获取输入张量的形状
        input_shape = input_ids.shape
        # 将输入张量重塑为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用embed_tokens方法对输入进行嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        # 如果存在project_in属性，则对inputs_embeds进行投影变换
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # 使用embed_positions方法对位置编码进行嵌入
        positions = self.embed_positions(position_ids)

        # 将输入嵌入和位置编码相加得到隐藏状态
        hidden_states = inputs_embeds + positions

        # 在Transformer层中进行前向传播
        hidden_state, all_hidden_states, attentions = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 如果存在final_layer_norm属性，则对最终的隐藏状态进行Layer Norm
        if self.final_layer_norm is not None:
            hidden_state = self.final_layer_norm(hidden_state)

        # 如果存在project_out属性，则对最终的隐藏状态进行投影变换
        if self.project_out is not None:
            hidden_state = self.project_out(hidden_state)

        # 如果需要输出所有隐藏状态
        if output_hidden_states:
            # 将当前隐藏状态添加到所有隐藏状态中
            all_hidden_states += (hidden_state,)

        # 将隐藏状态、所有隐藏状态和注意力权重作为输出
        outputs = [hidden_state, all_hidden_states, attentions]

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 将输出中不为None的部分作为元组返回
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=attentions,
        )
class FlaxOPTPreTrainedModel(FlaxPreTrainedModel):
    # OPT 模型的预训练模型类，继承自 FlaxPreTrainedModel
    config_class = OPTConfig
    # 使用 OPTConfig 类作为配置类
    base_model_prefix: str = "model"
    # 基础模型前缀设置为 "model"
    module_class: nn.Module = None
    # 模块类默认为空

    def __init__(
        self,
        config: OPTConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 定义初始化函数，接受配置、输入形状、种子、数据类型、是否初始化等参数
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 使用模块类创建模块对象
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化模型权重函数，接受随机种子、输入形状和参数字典
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 根据输入形状创建全零输入张量
        attention_mask = jnp.ones_like(input_ids)
        # 创建与输入张量相同形状的全 1 注意力掩码张量
        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 创建位置张量，用于表示每个位置的索引

        params_rng, dropout_rng = jax.random.split(rng)
        # 分割随机种子以获取参数和 dropout 的随机种子
        rngs = {"params": params_rng, "dropout": dropout_rng}
        # 组合参数和 dropout 的随机种子

        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
        )
        # 初始化模块的权重和参数

        random_params = module_init_outputs["params"]
        # 获取随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 对参数字典解冻并扁平化
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        # 处理缺失的参数键，冻结并重构参数字典
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化缓存的函数，接受批大小和最大长度参数
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 创建全 1 输入张量
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        # 创建与输入张量相同形状的全 1 注意力掩码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        # 创建位置张量

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 初始化模块的缓存变量
        return unfreeze(init_variables["cache"])
        # 返回解冻后的缓存变量
    # 定义类的调用方法，接受多个参数，其中包括输入的标识符、注意力掩码、位置标识符等
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        params: dict = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dropout_rng: PRNGKey = None,
        deterministic: bool = True,
    ):
        # 设置是否输出注意力矩阵，默认与配置文件中的设置一致
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，默认与配置文件中的设置一致
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认与配置文件中的设置一致
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果注意力掩码为空，则设置为全1的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果位置标识符为空，则根据注意力掩码的累积和来生成位置标识符
        if position_ids is None:
            position_ids = (attention_mask.cumsum(axis=1) * attention_mask) - 1

        # 如果需要处理随机数生成器，则创建相关的随机数字典
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 构建输入字典，其中包括模型参数和可选的缓存
        inputs = {"params": params or self.params}

        # 如果传入了过去的键值对，则将缓存传递给模型，并标记为可变
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 调用模型的应用方法，传入输入和其他相关参数
        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
        )

        # 将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回模型输出
        return outputs
# 定义一个名为FlaxOPTModule的类，继承自nn.Module
class FlaxOPTModule(nn.Module):
    # 类属性config，类型为OPTConfig
    config: OPTConfig
    # 类属性dtype，类型为jnp.dtype，默认值为jnp.float32，用于计算的数据类型

    # 定义setup方法
    def setup(self):
        # 实例化FlaxOPTDecoder类，传入config和dtype参数，赋值给self.decoder
        self.decoder = FlaxOPTDecoder(self.config, dtype=self.dtype)

    # 定义_get_decoder_module方法
    def _get_decoder_module(self):
        # 返回self.decoder
        return self.decoder

    # 定义__call__方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache=False,
    ):
        # 调用self.decoder，传入各种参数，赋值给decoder_outputs
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        # 如果return_dict为False，则返回decoder_outputs
        if not return_dict:
            return decoder_outputs

        # 返回FlaxBaseModelOutput实例，包括last_hidden_state、hidden_states、attentions字段
        return FlaxBaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


# 从transformers.models.bart.modeling_flax_bart.FlaxBartModel中复制代码并修改为FlaxOPTModel
class FlaxOPTModel(FlaxOPTPreTrainedModel):
    # 类属性config，类型为OPTConfig
    config: OPTConfig
    # 类属性dtype，类型为jnp.dtype，默认值为jnp.float32，用于计算的数据类型
    # 类属性module_class，值为FlaxOPTModule类

# 调用append_call_sample_docstring函数，传入FlaxOPTModel、_CHECKPOINT_FOR_DOC、FlaxBaseModelOutput、_CONFIG_FOR_DOC参数

# 调用add_start_docstrings函数，传入字符串和OPT_START_DOCSTRING参数
# 定义一个名为FlaxOPTForCausalLMModule的类，继承自nn.Module
class FlaxOPTForCausalLMModule(nn.Module):
    # 类属性config，类型为OPTConfig
    config: OPTConfig
    # 类属性dtype，类型为jnp.dtype，默认值为jnp.float32

    # 定义setup方法
    def setup(self):
        # 实例化FlaxOPTModule类，传入config和dtype参数，赋值给self.model
        self.model = FlaxOPTModule(config=self.config, dtype=self.dtype)
        # 实例化nn.Dense类，传入参数，赋值给self.lm_head
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    # 定义__call__方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        # 调用模型进行推理，传入输入的 token IDs、注意力掩码、位置编码等参数
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            init_cache=init_cache,  # 如果有缓存，则初始化缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回结果
            deterministic=deterministic,  # 是否确定性计算（不随机化）
        )

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]

        # 如果模型配置中指定词嵌入权重共享
        if self.config.tie_word_embeddings:
            # 获取共享的词嵌入矩阵
            shared_embedding = self.model.variables["params"]["decoder"]["embed_tokens"]["embedding"]
            # 对隐藏状态应用语言模型头，使用共享的词嵌入权重进行计算
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 对隐藏状态应用语言模型头，使用单独的权重矩阵进行计算
            lm_logits = self.lm_head(hidden_states)

        # 如果不要求以字典形式返回结果
        if not return_dict:
            # 返回语言模型的预测结果及其他输出
            return (lm_logits,) + outputs[1:]

        # 返回以字典形式封装的 MaskedLM 输出，包括预测 logits、隐藏状态、注意力权重等
        return FlaxMaskedLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串的装饰器，描述在语言建模任务中使用线性层与输入嵌入层权重相关联的 OPT 模型
@add_start_docstrings(
    """
    OPT Model with a language modeling head on top (linear layer with weights tied to the input embeddings) e.g for
    autoregressive tasks.
    """,
    OPT_START_DOCSTRING,
)
# 定义 FlaxOPTForCausalLM 类，继承自 FlaxOPTPreTrainedModel 类
class FlaxOPTForCausalLM(FlaxOPTPreTrainedModel):
    # 指定模块类为 FlaxOPTForCausalLMModule
    module_class = FlaxOPTForCausalLMModule

    # 为生成准备输入数据的方法
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 通常需要将注意力遮罩中 x > input_ids.shape[-1] 和 x < cache_length 的位置设为0
        # 但由于解码器使用因果注意力遮罩，这些位置已经被屏蔽了
        # 因此，可以在这里创建一个固定的注意力遮罩，编译效率更高
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成输入数据的方法
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

# 添加调用示例文档字符串的方法
append_call_sample_docstring(
    FlaxOPTForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
)
```